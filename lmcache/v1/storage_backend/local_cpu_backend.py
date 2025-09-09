# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from collections import OrderedDict
from concurrent.futures import Future
from typing import TYPE_CHECKING, List, Optional, Tuple
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.cache_controller.message import KVAdmitMsg, KVEvictMsg
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.config import LMCacheEngineMetadata
from lmcache.v1.lookup_server import LookupServerInterface
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MixedMemoryAllocator,
    MemoryObjMetadata,
)
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
import zmq
from nixl._api import nixl_agent
import uuid
from dataclasses import dataclass
import msgpack
import time
import pickle
import math


if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)

class LocalCPUBackend(StorageBackendInterface):
    """
    The local cpu backend size is variable depending on how much free space is
    left in the allocator so we cannot use LRUEvictor().
    (max_local_cpu_size > 0 initializes the memory_allocator)
    Even if local_cpu is False (the hot_cache is not used), contains(),
    insert_key(), remove(), get_blocking(), get_keys(), and clear()
    are still callable by the storage manager.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        memory_allocator: MemoryAllocatorInterface,
        lookup_server: Optional[LookupServerInterface] = None,
        lmcache_worker: Optional["LMCacheWorker"] = None,
    ):
        self.hot_cache: OrderedDict[CacheEngineKey, MemoryObj] = OrderedDict()
        self.use_hot = config.local_cpu
        self.lookup_server = lookup_server
        self.memory_allocator = memory_allocator
        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.cpu_lock = threading.Lock()

        self.stream = torch.cuda.Stream()

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.usage = 0

        self.layerwise = config.use_layerwise
        self.enable_blending = config.enable_blending

        # NIXL_PUSH_START

        self.vllm_block_size = metadata.vllm_block_size
        self._use_flashinfer = metadata.use_flashinfer
        self.use_mla = metadata.use_mla
        self.tp_rank = metadata.tp_rank

        # HACK static calculation of token size in bytes
        shape = (2, 32, 1, 1024)
        dtype = torch.bfloat16
        self._nixl_chunk_size = self.vllm_block_size

        self.element_size = torch.empty((), dtype=dtype).element_size()
        self._nixl_block_size = (math.prod(shape)) * (torch.tensor([], dtype=dtype).element_size()) * self._nixl_chunk_size # XXXX fixme
        print(f"XXX nixl block size={self._nixl_block_size}")

        assert config.nixl_role in ["sender", "receiver"], (
            f"Invalid role: {config.nixl_role}, must be either "
            f"sender or receiver"
        )

        self._agent = nixl_agent(config.nixl_role)
        self._agent_gpu = nixl_agent("GPU")
        self.engine_id = metadata.engine_id
        self.kv_caches_base_addr: dict[str, list[int]] = {}

        self._nixl_role = config.nixl_role
        self._nixl_operation = config.nixl_operation

        # Nixl register memory
        mem_base_addr, mem_size = self.memory_allocator.get_mem_layout()
        self.mem_base_addr = mem_base_addr
        self.mem_size = mem_size

        local_mem = [(mem_base_addr, mem_size, 0, "")]
        descs = self._agent.get_reg_descs(local_mem, "DRAM")
        self._agent.register_memory(descs)
        print(f"XXX register descs= {local_mem} {mem_base_addr} : {mem_size}")

        # Register local/src descr for NIXL xfer.
        assert mem_size % self._nixl_block_size == 0
        self.src_xfer_side_handle = self.create_xfer_descs("NIXL_INIT_AGENT", mem_base_addr, mem_size // self._nixl_block_size, self._nixl_block_size)

        if config.nixl_role == "sender":
            print(f"XXXX SENDER")

            self._running = True
            # Start Nixl completion thread
            self._transfers = {}
            self._transfers_lock = threading.Lock()
            self._transfers_thread = threading.Thread(
                target=self._send_transfers_loop, daemon=True
            )
            self._transfers_thread.start()

            local_meta = self._agent.get_agent_metadata()

            # Initialize the ZeroMQ context and side channel
            self._context = zmq.Context()  # type: ignore
            # Change from PAIR to DEALER socket
            self._side_channel = self._context.socket(zmq.DEALER)  # type: ignore
            # Set an identity for this DEALER socket
            self._side_channel.setsockopt(
                zmq.IDENTITY,  # type: ignore
                f"sender-{uuid.uuid4().hex}".encode(),
            )  # type: ignore
            worker_id = 0 # HACK for now
            self._side_channel.connect("tcp://{}:{}".format(config.nixl_receiver_host, config.nixl_receiver_port + worker_id))
            self._side_channel.setsockopt(zmq.LINGER, 0)  # type: ignore

            message = (local_meta, mem_base_addr, mem_size // self._nixl_block_size, self._nixl_block_size)
            data = pickle.dumps(message)

            self._side_channel.send(data)
            msg = self._side_channel.recv()
            remote_meta, remote_mem_base_addr, remote_num_blocks, remote_block_size = pickle.loads(msg)
            assert self._nixl_block_size == remote_block_size
            self.peer_name = self._agent.add_remote_agent(remote_meta).decode("utf-8")

            # prepare descriptors in case we do WRITE
            self.dst_xfer_side_handle = self.create_xfer_descs(self.peer_name, remote_mem_base_addr, remote_num_blocks, remote_block_size)
            print(f"SENDER end handshake with {self.peer_name}")
        else:
            print(f"XXX RECEIVER")

            # Initialize the ZeroMQ context and side channel
            self._context = zmq.Context()  # type: ignore
            # Change from PAIR to ROUTER socket
            self._side_channel = self._context.socket(zmq.ROUTER)  # type: ignore
            worker_id = 0 # HACK for now
            self._side_channel.bind(
                "tcp://{}:{}".format(config.nixl_receiver_host, config.nixl_receiver_port + worker_id)
            )
            self._side_channel.setsockopt(zmq.LINGER, 0)  # type: ignore
            # Add a timeout for the side channel
            self._side_channel.setsockopt(
                zmq.RCVTIMEO,  # type: ignore
                5000,  # Set a timeout for receiving to avoid blocking
            )

            self._running = True
            # Start Nixl completion thread
            self._completed_h2h = {}
            self._req_blocks = {}
            self._transfers = {}
            self._transfers_lock = threading.Lock()
            self._transfers_thread = threading.Thread(
                target=self._recv_transfers_loop, daemon=True
            )
            self._transfers_thread.start()

            # Start the receiver thread
            self._receiver_thread = threading.Thread(
                target=self._receiver_loop, daemon=True
            )
            self._sender_id = None
            self._receiver_thread.start()

        # NIXL_PUSH_END

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in nixl."""

        _, first_kv_cache = next(iter(kv_caches.items()))
        kv_elem_size = first_kv_cache.element_size()

        if self._use_flashinfer:
            # FlashInfer swaps 2<->num_blocks dimensions.
            self.num_blocks = first_kv_cache.shape[0]
            block_rank = 4  # [2, block_size, kv_heads, head_dim]
        else:
            self.num_blocks = first_kv_cache.shape[1]
            block_rank = 3  # [block_size, kv_heads, head_dim]
        block_shape = first_kv_cache.shape[-block_rank:]
        block_size, n_kv_heads, head_dim = block_shape[-3:]
        # head size in bytes.
        self.slot_size_bytes = kv_elem_size * n_kv_heads * head_dim
        assert block_size == self.vllm_block_size
        # TODO(tms): self.block_len needs to be per-layer for sliding window,
        # hybrid attn, etc
        # block size in bytes
        self.block_len = kv_elem_size * math.prod(block_shape)

        logger.info(
            "XXXXX Registering KV_Caches: use_mla: %s, num_blocks: %s, "
            "block_shape: %s, per_layer_kv_cache_shape: %s", self.use_mla,
            self.num_blocks, block_shape, first_kv_cache.shape)

        self.kv_caches = kv_caches
        kv_caches_base_addr = []
        caches_data = []

        # Note(tms): I modified this from the original region setup code.
        # K and V are now in different regions. Advantage is that we can
        # elegantly support MLA and any cases where the K and V tensors
        # are non-contiguous (it's not locally guaranteed that they will be)
        # Disadvantage is that the encoded NixlAgentMetadata is now larger
        # (roughly 8KB vs 5KB).
        # Conversely for FlashInfer, K and V are transferred in the same tensor
        # to better exploit the memory layout (ie num_blocks is the first dim).
        for cache_or_caches in kv_caches.values():
            # Normalize to always be a list of caches
            cache_list = [cache_or_caches] if self.use_mla or self._use_flashinfer \
                else cache_or_caches
            for cache in cache_list:
                base_addr = cache.data_ptr()
                region_len = self.num_blocks * self.block_len
                caches_data.append(
                    (base_addr, region_len, cache.device.index, ""))
                kv_caches_base_addr.append(base_addr)
        self.kv_caches_base_addr[self.engine_id] = kv_caches_base_addr
        self.num_regions = len(caches_data)
        self.num_layers = len(self.kv_caches.keys())

        # TODO(mgoin): remove this once we have hybrid memory allocator
        # Optimization for models with local attention (Llama 4)
        # if self.vllm_config.model_config.hf_config.model_type == "llama4":
        #     from transformers import Llama4TextConfig
        #     assert isinstance(self.vllm_config.model_config.hf_text_config,
        #                       Llama4TextConfig)
        #     llama4_config = self.vllm_config.model_config.hf_text_config
        #     no_rope_layers = llama4_config.no_rope_layers
        #     chunk_size = llama4_config.attention_chunk_size
        #     chunk_block_size = math.ceil(chunk_size / self.vllm_block_size)
        #     for layer_idx in range(self.num_layers):
        #         # no_rope_layers[layer_idx] == 0 means NoPE (global)
        #         # Any other value means RoPE (local chunked)
        #         is_local_attention = no_rope_layers[layer_idx] != 0
        #         block_window = chunk_block_size if is_local_attention else None
        #         self.block_window_per_layer.append(block_window)
        #     logger.debug("Llama 4 block window per layer mapping: %s",
        #                  self.block_window_per_layer)
        #     assert len(self.block_window_per_layer) == self.num_layers

        descs = self._agent_gpu.get_reg_descs(caches_data, "VRAM")
        logger.debug("XXX Registering descs: %s", caches_data)
        self._agent_gpu.register_memory(descs)

        # Register local/src descr for NIXL xfer.
        blocks_data = []
        for base_addr in self.kv_caches_base_addr[self.engine_id]:
            # NOTE With heter-TP, more blocks are prepared than what are
            # needed as self.num_blocks >= nixl_agent_meta.num_blocks. We
            # could create fewer, but then _get_block_descs_ids needs to
            # select agent_meta.num_blocks instead of self.num_blocks for
            # local descr, and that makes handling regular flow less clean.
            for block_id in range(self.num_blocks):
                block_offset = block_id * self.block_len
                addr = base_addr + block_offset
                # (addr, len, device id)
                blocks_data.append((addr, self.block_len, self.tp_rank))
        logger.info("XXXX Created %s blocks for src engine %s and rank %s",
                     len(blocks_data), self.engine_id, self.tp_rank)

        gpu_descs = self._agent_gpu.get_xfer_descs(blocks_data, "VRAM", is_sorted=True)

        # Excahnge gpu agent metadata and prepare xfer list
        gpu_meta = self._agent_gpu.get_agent_metadata()
        self.gpu_peer_name = self._agent.add_remote_agent(gpu_meta)
        print(f"XXXX {self.gpu_peer_name}\n {gpu_descs}")

        self.remote_gpu_xfer_handle = self._agent.prep_xfer_dlist(self.gpu_peer_name, gpu_descs, "VRAM")
        assert self.remote_gpu_xfer_handle != 0

        self.src_xfer_block_side_handle = self.create_xfer_descs("NIXL_INIT_AGENT", self.mem_base_addr, self.mem_size // self.block_len, self.block_len)
        assert self.src_xfer_block_side_handle != 0

        print(f"XXXX {self.mem_base_addr} cpu blocks {self.mem_size //self.block_len} block len {self.block_len}\n {self.src_xfer_block_side_handle}\n {self.remote_gpu_xfer_handle}")

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        return None, None

    def _send_transfers_loop(self):
        while self._running:

            remove_handles = []
            if self._nixl_operation == "WRITE":
                with self._transfers_lock:
                    for handle, (start_token, end_token, t_done, msg_size, start, req_id, memory_objs) in self._transfers.items(): # XXX temp - need to use get_finished() here
                        state = self._agent.check_xfer_state(handle)

                        if state == "ERR":
                            print("Transfer got to Error state.")
                            exit()
                        elif state == "DONE":
                            end = time.perf_counter()
                            logger.info(f"========== TRANSFER completed:  {msg_size/(1<<20):.2f} MB BW: {msg_size/((end - start) * (1 << 30)):.3f} GB/s")

                            self._agent.release_xfer_handle(handle)
                            remove_handles.append(handle)

                    for handle in remove_handles:
                        assert handle in self._transfers
                        del self._transfers[handle]

            time.sleep(0.001)  # Avoid busy waitingsleep

    def get_offset(self, kv_shape, kv, layers, tokens, dim):
        x,y,z,w = kv_shape
        strides = (y*z*w, z*w, w, 1)

        return kv*strides[0] + layers*strides[1] + tokens*strides[2] + dim*strides[3]

    def _get_mem_cpu_desc_ids(self,
                              memory_objs: list[MemoryObj]) -> Tuple[List[int], int]:
        descs_ids: list[int] = []
        n_blocks = 0
        kv_shape = None

        for mem_obj in memory_objs:
            base_offset = mem_obj.metadata.address
            kv_shape = mem_obj.get_shape()
            for layer in range(kv_shape[1]):
                for block in range(0, kv_shape[2], self.vllm_block_size):
                    for kv in range(kv_shape[0]):
                        idx = (kv, layer, block, 0)
                        offset = self.get_offset(kv_shape, kv, layer, block, 0) * self.element_size
                        offset += base_offset
                        assert offset % self.block_len == 0
                        descs_ids.append(offset // self.block_len)

        assert len(descs_ids) % (kv_shape[0] * kv_shape[1]) == 0
        n_blocks = len(descs_ids) // (kv_shape[0] * kv_shape[1])

        return descs_ids, n_blocks

    def _get_gpu_descs_ids(self,
                           block_ids: list[int]) -> List[int]:
        """
        Get the descs ids for a set of block ids.
        """
        region_ids = range(self.num_regions)

        num_blocks = self.num_blocks

        # Compute the desc ids for each block.
        descs_ids: List[int] = []
        for reg_id in region_ids:
            for block_id in block_ids:
                descs_ids.append(reg_id * num_blocks + block_id)
        return descs_ids

    def _h2d_transfer(self, req_id: str, memory_objs: list[MemoryObj],
                      gpu_block_ids: list[int]) -> List[int]:
        cpu_desc_ids, n_blocks = self._get_mem_cpu_desc_ids(memory_objs)
        current_gpu_block_ids = gpu_block_ids[:n_blocks]
        next_gpu_block_ids = gpu_block_ids[n_blocks:]

        logger.info(f"XXXX {current_gpu_block_ids}\n {next_gpu_block_ids}")
        gpu_desc_ids = self._get_gpu_descs_ids(current_gpu_block_ids)
        logger.info(f"XXX write to gpu: cpu blocks={len(cpu_desc_ids)} gpu blocks={len(gpu_desc_ids)}")
        start = time.perf_counter()

        handle = self._agent.make_prepped_xfer(
            "WRITE",
            self.src_xfer_block_side_handle,
            cpu_desc_ids,
            self.remote_gpu_xfer_handle,
            gpu_desc_ids,
            notif_msg=b"XXXX",
            skip_desc_merge=False,
        )

        self._agent.transfer(handle)

        sender_done = False
        while not sender_done:
            state = self._agent.check_xfer_state(handle)
            if state == "ERR":
                print("Transfer got to Error state.")
                exit()
            elif state == "DONE":
                self._agent.release_xfer_handle(handle)
                sender_done = True

            time.sleep(0.001)  # Avoid busy waitingsleep

        return next_gpu_block_ids

    def _recv_transfers_loop(self):
        while self._running:

            completed = []
            if self._nixl_operation == "READ":
                with self._transfers_lock:
                    for handle, (start_token, end_token, t_done, msg_size, start, req_id, memory_objs) in self._transfers.items():
                        state = self._agent.check_xfer_state(handle)

                        if state == "ERR":
                            print("Transfer got to Error state.")
                            exit()
                        elif state == "DONE":
                            end = time.perf_counter()
                            logger.info(f"========== TRANSFER completed:  {msg_size/(1<<20):.2f} MB BW: {msg_size/((end - start) * (1 << 30)):.3f} GB/s")

                            self._agent.release_xfer_handle(handle)
                            t_done.set()
                            if req_id not in self._completed_h2h:
                                self._completed_h2h[req_id] = []
                            if len(self._completed_h2h[req_id]) > 0:
                                l_start_token, l_end_token, *_ = self._completed_h2h[req_id][-1]
                                assert l_end_token  == start_token, f"Error: {l_end_token} != {start_token}"
                            self._completed_h2h[req_id].append((start_token, end_token, msg_size, memory_objs))
                            completed.append((handle, msg_size, req_id))

                    for handle, msg_size, req_id in completed:
                        assert handle in self._transfers
                        del self._transfers[handle]

            else:
                # WRITE
                all_notifs = self._agent.get_new_notifs().values() # XXX the code assumes a single peer for now
                for notifs in all_notifs:
                    for notif in notifs:
                        handle = notif.decode("utf-8")

                        value = None
                        with self._transfers_lock:
                            value = self._transfers.pop(handle, None)

                        if value is not None:
                            start_token, end_token, t_done, msg_size, start, req_id, memory_objs = value
                            end = time.perf_counter()
                            logger.info(f"========== TRANSFER completed:  {msg_size/(1<<20):.2f} MB BW: {msg_size/((end - start) * (1 << 30)):.3f} GB/s")

                            t_done.set()
                            if req_id not in self._completed_h2h:
                                self._completed_h2h[req_id] = []
                            if len(self._completed_h2h[req_id]) > 0:
                                l_start_token, end_token, *_ = self._completed_h2h[req_id][-1]
                                assert end_token == start_token, f"Error: {end_token} != {start_token}"
                            self._completed_h2h[req_id].append((start_token, end_token, msg_size, memory_objs))

            completed = []
            ready_to_xfer = False
            req_id = None
            gpu_block_ids = None
            with self._transfers_lock:
                for req_id, gpu_block_ids in self._req_blocks.items():
                    if req_id in self._completed_h2h:
                        ready_to_xfer = True
                        completed.append(req_id)
                        break

                if ready_to_xfer:
                    logger.info(f"XXX Start h2d transfer req={req_id} blocks={gpu_block_ids} size={msg_size}")
                    msg_list = self._completed_h2h.pop(req_id, None)

                    memory_objs: List[MemoryObj] = []
                    for msg in msg_list:
                        logger.info(f"XXXX extend memory_objs {len(memory_objs)}")
                        memory_objs.extend(msg[3])

                    next_gpu_block_ids = self._h2d_transfer(req_id, memory_objs, gpu_block_ids)
                    self._req_blocks.pop(req_id, None)
                    if len(next_gpu_block_ids) > 0:
                        self._req_blocks[req_id] = next_gpu_block_ids

            time.sleep(0.001)  # Avoid busy waiting

    def insert_transfer(self, handle, start_token, end_token, msg_size, start, req_id, memory_objs):
        with self._transfers_lock:
            self._transfers[handle] = (start_token, end_token, threading.Event(), msg_size, start, req_id, memory_objs)

    def wait_for_transfer(self, handle):
        t_done = None

        with self._transfers_lock:
            if handle in self._transfers:
                t_done = self._transfers[handle][2]

        logger.debug(f"XXX wait_for {handle} {t_done}")
        if t_done:
            t_done.wait()
        logger.debug(f"XXX wait_for completed {handle}")

    def _receiver_loop(self):
        poller = zmq.Poller()  # type: ignore
        poller.register(self._side_channel, zmq.POLLIN)  # type: ignore
        # Use a shorter timeout to be more responsive to shutdown
        POLL_TIMEOUT_MS = 1000  # 1s timeout

        local_meta = self._agent.get_agent_metadata()

        while self._running:
            try:
                # Wait for a request from the side channel with shorter timeout
                evts = poller.poll(timeout=POLL_TIMEOUT_MS)
                if not evts:
                    continue

                sender_id, msg = self._side_channel.recv_multipart()
                if not msg:
                    logger.warn("Received empty message on the side channel")
                    time.sleep(0.1)  # Avoid busy waiting
                    continue

                # New sender connection
                if not self._sender_id:
                    self._sender_id  = sender_id  # HACK single sender for now
                    sender_meta, sender_mem_base_addr, sender_num_blocks, sender_block_size = pickle.loads(msg)
                    assert self._nixl_block_size == sender_block_size
                    #sender_meta = pickle.loads(msg)
                    #sender_meta = msg
                    # Now, msg should be the sender metadata
                    # Initialize a new pipe for this sender
                    assert sender_meta is not None, (
                        "The sender_meta should be provided on the receiver side"
                    )
                    self.peer_name = self._agent.add_remote_agent(sender_meta).decode("utf-8")

                    self.dst_xfer_side_handle = self.create_xfer_descs(self.peer_name, sender_mem_base_addr, sender_num_blocks, sender_block_size)

                    # prepare message to sender
                    mem_base_addr, mem_size = self.memory_allocator.get_mem_layout()
                    message = (local_meta, mem_base_addr, mem_size // self._nixl_block_size, self._nixl_block_size)
                    data = pickle.dumps(message)

                    self._side_channel.send_multipart([sender_id, data])
                    print(f"XXX RECEIVER end handshake")
                    logger.info(f"New sender connected with ID: {sender_id.decode()}")
                    continue

                #request = NixlRequest.deserialize(msg)

                req_id, start_token, end_token, keys, metadatas, nixl_operation = pickle.loads(msg)
                logger.info(f"XXX Received request {req_id} for NIXL {nixl_operation} with {len(keys)}:{len(metadatas)} from sender {sender_id.decode()}")

                l_metadatas = []
                memory_objs = []
                local_descs_ids = []
                remote_descs_ids = []
                total_size = 0
                for key, meta in zip(keys, metadatas, strict=False):
                    mem_obj = self.allocate(meta.shape, meta.dtype)
                    memory_objs.append(mem_obj)
                    l_metadatas.append(mem_obj.metadata)

                    assert meta.phy_size % self._nixl_block_size == 0
                    total_size += meta.phy_size
                    num_blocks = meta.phy_size // self._nixl_block_size
                    assert mem_obj.metadata.address % self._nixl_block_size == 0
                    local_base_block_id = mem_obj.metadata.address // self._nixl_block_size
                    remote_base_block_id = meta.address // self._nixl_block_size

                    if nixl_operation == "READ":
                        for block_id in range(num_blocks):
                            local_descs_ids.append(local_base_block_id + block_id)
                            remote_descs_ids.append(remote_base_block_id + block_id)

                handle = None
                if nixl_operation == "READ":
                    # Prepare transfer with Nixl.
                    start = time.perf_counter()

                    handle = self._agent.make_prepped_xfer(
                        "READ",
                        self.src_xfer_side_handle,
                        local_descs_ids,
                        self.dst_xfer_side_handle,
                        remote_descs_ids,
                        notif_msg="XXX",
                        skip_desc_merge=False,
                    )

                    for mem_obj in memory_objs:
                        mem_obj.metadata.handle = handle

                    # Begin async xfer.
                    self.insert_transfer(handle, start_token, end_token, total_size, start, req_id, memory_objs)

                    self._agent.transfer(handle)
                    self.batched_submit_put_task([start_token], [end_token], keys, memory_objs, req_id)

                    for memory_obj in memory_objs:
                        memory_obj.ref_count_down()
                else: # WRITE
                    handle = str(uuid.uuid4())

                    for mem_obj in memory_objs:
                        mem_obj.metadata.handle = handle

                    start = time.perf_counter()
                    self.insert_transfer(handle, start_token, end_token, total_size, start, req_id, memory_objs)

                    self.batched_submit_put_task([start_token], [end_token], memory_objs, req_id)

                    message = (l_metadatas, handle)
                    data = pickle.dumps(message)

                    self._side_channel.send_multipart([sender_id, data])
                    logger.debug(f"XXX Receiver sent to sender metadata len={len(l_metadatas)} for {handle}")

            except zmq.Again as e:  # type: ignore
                # Handle the timeout when waiting for a message
                logger.debug(
                    "Timeout waiting for a message on the side channel: %s",
                    str(e),
                )
                continue
            except Exception as e:
                logger.error("Failed to process receiver loop: %s", str(e))
                if self._running:
                    time.sleep(0.01)

    def create_xfer_descs(self, agent_name, base_addr, num_blocks, block_len):
        blocks_data = []
        logger.info(f"XXXX {agent_name}::n_blocks={num_blocks} block_len={block_len}")
        for block_id in range(num_blocks):
            block_offset = block_id * block_len
            addr = base_addr + block_offset
            blocks_data.append((addr, block_len, 0))

        #descs = self._agent.get_xfer_descs(blocks_data, "DRAM")

        return self._agent.prep_xfer_dlist(agent_name, blocks_data, "DRAM", True)

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return False
            if pin:
                self.hot_cache[key].pin()
            return True

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        contains() and exists_in_put_tasks() should be checked together
        """
        return False

    def submit_put_task(
        self, key: CacheEngineKey, memory_obj: MemoryObj
    ) -> Optional[Future]:
        """
        Synchronously put the MemoryObj into the local cpu backend.
        """
        with self.cpu_lock:
            if key in self.hot_cache:
                old_memory_obj = self.hot_cache.pop(key)
                old_memory_obj.ref_count_down()
            self.hot_cache[key] = memory_obj
            memory_obj.ref_count_up()

            self.usage += memory_obj.get_size()
            self.stats_monitor.update_local_cache_usage(self.usage)

            # TODO(Jiayi): optimize this with batching?
            # push kv admit msg
            if self.lmcache_worker is not None:
                self.lmcache_worker.put_msg(
                    KVAdmitMsg(self.instance_id, key.worker_id, key.chunk_hash, "cpu")
                )
        return None

    def retrieve_async(
        self,
        req_id: str,
        local_block_ids: list[int],
    ) -> None:
        with self._transfers_lock:
            self._req_blocks[req_id] = local_block_ids

    def batched_submit_put_task(
        self,
        starts: List[int],
        ends: List[int],
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
        req_id,
    ) -> Optional[List[Future]]:
        """
        Synchronously put the MemoryObjs into the local cpu backend.
        """
        if not self.use_hot:
            return None

        # NIXL PUSH START

        # TODO(Jiayi): optimize this with batching
        metadatas = []
        pushed_keys = []
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            if memory_obj.get_shape()[2] % self._nixl_chunk_size == 0:
                metadatas.append(memory_obj.metadata)
                pushed_keys.append(key)
            else:
                logger.warning(f"XXX skip not aligned chunk size {memory_obj.get_shape()[2]}")
            self.submit_put_task(key, memory_obj)

        if self._nixl_role == "sender":
            # REMOVE request = NixlRequest(keys=keys, metadatas=metadatas)
            message = (req_id, starts[0], ends[-1], pushed_keys, metadatas, self._nixl_operation)
            data = pickle.dumps(message)

            self._side_channel.send(data)

            if self._nixl_operation == "WRITE":
                msg = self._side_channel.recv()
                r_metadatas, msg_id = pickle.loads(msg)

                logger.debug(f"XXX Sender got metadata for {msg_id} descriptor length {len(r_metadatas)} metadatas {len(metadatas)}")
                assert len(metadatas) == len(r_metadatas)

                local_descs_ids = []
                remote_descs_ids = []
                total_size = 0
                for l_meta, r_meta in zip(metadatas, r_metadatas, strict=False):
                    assert r_meta.phy_size % self._nixl_block_size == 0
                    num_blocks = r_meta.phy_size // self._nixl_block_size
                    total_size += r_meta.phy_size

                    local_base_block_id = l_meta.address // self._nixl_block_size
                    remote_base_block_id = r_meta.address // self._nixl_block_size

                    for block_id in range(num_blocks):
                        local_descs_ids.append(local_base_block_id + block_id)
                        remote_descs_ids.append(remote_base_block_id + block_id)

                # Prepare transfer with Nixl.
                start = time.perf_counter()

                handle = self._agent.make_prepped_xfer(
                    "WRITE",
                    self.src_xfer_side_handle,
                    local_descs_ids,
                    self.dst_xfer_side_handle,
                    remote_descs_ids,
                    notif_msg=msg_id,
                    skip_desc_merge=False,
                )

                # XXX TODO: Increase reference count of memory objects till transfer is completed

                self.insert_transfer(handle, starts[0], ends[-1], total_size, start, req_id, None)

                # Begin async xfer.
                self._agent.transfer(handle)

                # sender_done = False
                # while not sender_done:
                #     state = self._agent.check_xfer_state(handle)
                #     if state == "ERR":
                #         print("Transfer got to Error state.")
                #         exit()
                #     elif state == "DONE":
                #         print(f"XXX write DONE")
                #         sender_done = True
                #     time.sleep(0.001)  # Avoid busy waitingsleep


            #TODO: These memory buffers should be pinned in memory till we get notif for completion

            # notifs = self._agent.get_new_notifs()

            # while len(notifs) == 0:
            #     notifs = self._agent.get_new_notifs()

            # assert len(notifs) == 1, f"notifs len error = {len(notifs)}"

        # NIXL PUSH END

        return None

    # NOTE (Jiayi): prefetch might be deprecated in the future.
    # Should be replaced by `move`.
    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        return None

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return None
            memory_obj = self.hot_cache[key]
            # ref count up for caller to avoid situation where the memory_obj
            # is evicted from the local cpu backend before the caller calls
            # ref count up themselves
            memory_obj.ref_count_up()

            handle = memory_obj.metadata.handle
            if handle is not None:
                self.wait_for_transfer(handle)
                memory_obj.metadata.handle = None

            self.hot_cache.move_to_end(key)
            return memory_obj

    def get_non_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        """
        Return the dummy future object.
        """
        with self.cpu_lock:
            if key not in self.hot_cache:
                return None
            memory_obj = self.hot_cache[key]
            memory_obj.ref_count_up()
            self.hot_cache.move_to_end(key)
            f: Future = Future()
            f.set_result(memory_obj)
            return f

    def pin(self, key: CacheEngineKey) -> bool:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return False
            memory_obj = self.hot_cache[key]
            memory_obj.pin()
            return True

    def unpin(self, key: CacheEngineKey) -> bool:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return False
            memory_obj = self.hot_cache[key]
            memory_obj.unpin()
            return True

    def remove(self, key: CacheEngineKey, free_obj=True) -> bool:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return False
            memory_obj = self.hot_cache.pop(key)
            if free_obj:
                memory_obj.ref_count_down()

            self.usage -= memory_obj.get_size()
            self.stats_monitor.update_local_cache_usage(self.usage)

            if self.lmcache_worker is not None:
                self.lmcache_worker.put_msg(
                    KVEvictMsg(self.instance_id, key.worker_id, key.chunk_hash, "cpu")
                )
            # NOTE (Jiayi): This `return True` might not accurately reflect
            # whether the key is removed from the actual memory because
            # other backends might still (temporarily) hold the memory object.
            return True

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: Optional[MemoryFormat] = None,
        eviction: bool = True,
    ) -> Optional[MemoryObj]:
        """
        Allocate a memory object of shape and dtype
        evict if necessary. Storage manager should always call
        local_cpu_backend.allocate() to get memory objects
        regardless of whether local_cpu is True or False
        """
        if fmt is None:
            if self.layerwise:
                if self.enable_blending:
                    fmt = MemoryFormat.KV_2TD
                else:
                    fmt = MemoryFormat.KV_T2D
            else:
                fmt = MemoryFormat.KV_2LTD

        memory_obj = self.memory_allocator.allocate(shape, dtype, fmt)
        if memory_obj is not None or not eviction:
            return memory_obj

        assert isinstance(self.memory_allocator, MixedMemoryAllocator)

        evict_keys = []
        with self.cpu_lock:
            for evict_key in self.hot_cache:
                old_mem_obj = self.hot_cache[evict_key]
                # If the ref_count > 1, we cannot evict it as the cpu memory
                # might be used as buffers by other storage backends
                # Also, don't evict pinned objects
                if old_mem_obj.get_ref_count() > 1 or old_mem_obj.is_pinned:
                    continue
                evict_keys.append(evict_key)

                old_mem_obj.ref_count_down()
                memory_obj = self.memory_allocator.allocate(shape, dtype, fmt)
                logger.debug("Evicting 1 chunk from cpu memory")
                if memory_obj is not None:
                    break
        for evict_key in evict_keys:
            # already freed above in order to allocate new memory object
            # this is to remove the key from the hot cache
            self.remove(evict_key, free_obj=False)
        if self.lookup_server is not None:
            self.lookup_server.batched_remove(evict_keys)
        return memory_obj

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        batch_size: int,
        fmt: Optional[MemoryFormat] = None,
        eviction: bool = True,
    ) -> Optional[List[MemoryObj]]:
        """
        Batched allocate `batch_size` memory objects of shape and dtype
        evict if necessary. Storage manager should always call
        local_cpu_backend.allocate() to get memory objects
        regardless of whether local_cpu is True or False
        """
        if fmt is None:
            if self.layerwise:
                if self.enable_blending:
                    fmt = MemoryFormat.KV_2TD
                else:
                    fmt = MemoryFormat.KV_T2D
            else:
                fmt = MemoryFormat.KV_2LTD

        memory_objs = self.memory_allocator.batched_allocate(
            shape, dtype, batch_size, fmt
        )
        if memory_objs is not None or not eviction:
            return memory_objs

        assert isinstance(self.memory_allocator, MixedMemoryAllocator)

        # NOTE: Tune this number for performance.
        # Setting it to small will cause more eviction overhead.
        # Setting it to large might result in lower cache hit
        # because more caches are evicted.
        # blocks_to_free = batch_size

        evict_keys = []
        old_mem_objs = []
        with self.cpu_lock:
            for evict_key in self.hot_cache:
                if evict_key in evict_keys:
                    continue
                old_mem_obj = self.hot_cache[evict_key]
                # If the ref_count > 1, we cannot evict it as the cpu memory
                # might be used as buffers by other storage backends
                # Also, don't evict pinned objects
                if old_mem_obj.get_ref_count() > 1 or old_mem_obj.is_pinned:
                    continue
                # HACK: We assume batch_size=num_layers here.
                # We also assume if the one layer's ref_count > 1 or pinned,
                # then the other layers are also ref_count > 1 or
                # pinned in the cpu memory.
                evict_key_all_layer = evict_key.split_layers(batch_size)
                evict_keys.extend(evict_key_all_layer)
                for key in evict_key_all_layer:
                    old_mem_objs.append(self.hot_cache[key])

                # if len(old_mem_objs) < blocks_to_free:
                #    continue

                self.memory_allocator.batched_free(old_mem_objs)
                memory_objs = self.memory_allocator.batched_allocate(
                    shape, dtype, batch_size, fmt
                )

                logger.debug(f"Evicting {len(old_mem_objs)} chunks from cpu memory")

                if memory_objs is not None:
                    break
                old_mem_objs = []
        for evict_key in evict_keys:
            # already freed above in order to allocate new memory objects
            # this is to remove the key from the hot cache
            self.remove(evict_key, free_obj=False)
        if self.lookup_server is not None:
            self.lookup_server.batched_remove(evict_keys)
        return memory_objs

    def write_back(self, key: CacheEngineKey, memory_obj: MemoryObj):
        if memory_obj is None or not self.use_hot:
            return

        if memory_obj.tensor is not None and memory_obj.tensor.is_cuda:
            self.cpu_lock.acquire()
            if key in self.hot_cache:
                self.cpu_lock.release()
                return
            self.cpu_lock.release()

            # Allocate a cpu memory object
            cpu_memory_obj = self.memory_allocator.allocate(
                memory_obj.get_shape(),
                memory_obj.get_dtype(),
                fmt=memory_obj.get_memory_format(),
            )

            if cpu_memory_obj is None:
                logger.warning("Memory allocation failed in cachegen deserializer")
                return None

            # Copy the tensor to the cpu memory object
            assert cpu_memory_obj.tensor is not None
            self.stream.wait_stream(torch.cuda.default_stream())
            with torch.cuda.stream(self.stream):
                cpu_memory_obj.tensor.copy_(memory_obj.tensor, non_blocking=True)
            memory_obj.tensor.record_stream(self.stream)

            # Update the hot cache
            self.cpu_lock.acquire()
            self.hot_cache[key] = cpu_memory_obj
            cpu_memory_obj.ref_count_up()
            self.cpu_lock.release()

            # Push kv msg
            if self.lmcache_worker is not None:
                self.lmcache_worker.put_msg(
                    KVAdmitMsg(self.instance_id, key.worker_id, key.chunk_hash, "cpu")
                )

            logger.debug("Updated hot cache!")
        else:
            self.cpu_lock.acquire()
            if self.use_hot and key not in self.hot_cache:
                self.hot_cache[key] = memory_obj
                memory_obj.ref_count_up()
                self.cpu_lock.release()

                # Push kv msg
                if self.lmcache_worker is not None:
                    self.lmcache_worker.put_msg(
                        KVAdmitMsg(
                            self.instance_id,
                            key.worker_id,
                            key.chunk_hash,
                            "cpu",
                        )
                    )
            else:
                self.cpu_lock.release()

    def get_keys(self) -> List[CacheEngineKey]:
        """
        array ordering of keys from LRU to MRU
        """
        with self.cpu_lock:
            return list(self.hot_cache.keys())

    def clear(self) -> int:
        """
        counts the number of memory objects removed
        """
        if not self.use_hot:
            return 0
        clear_keys = []
        with self.cpu_lock:
            for key in self.hot_cache:
                memory_obj = self.hot_cache[key]
                if memory_obj.get_ref_count() > 1:
                    continue
                clear_keys.append(key)

        for key in clear_keys:
            self.remove(key)

        if self.lookup_server is not None:
            self.lookup_server.batched_remove(clear_keys)

        return len(clear_keys)

    def close(self) -> None:
        self.clear()

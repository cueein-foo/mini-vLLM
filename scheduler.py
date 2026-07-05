"""
Mini vLLM Scheduler Implementation
A simplified version of vLLM's scheduling logic for LLM inference
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum
import time


class RequestStatus(Enum):
    """Status of a request in the scheduling queue"""
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"


@dataclass
class SchedulingRequest:
    """Represents a request to be scheduled"""
    request_id: str
    prompt_length: int
    max_tokens: int
    priority: int = 0
    arrival_time: float = 0.0
    status: RequestStatus = RequestStatus.WAITING
    
    # Scheduling metadata
    allocated_blocks: int = 0
    computed_tokens: int = 0


class BlockManager:
    """Manages memory blocks for KV cache"""
    
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.allocated_blocks: Dict[str, List[int]] = {}
    
    def allocate_blocks(self, request_id: str, num_required: int) -> Optional[List[int]]:
        """Allocate blocks for a request"""
        if len(self.free_blocks) < num_required:
            return None
        
        allocated = self.free_blocks[:num_required]
        self.free_blocks = self.free_blocks[num_required:]
        self.allocated_blocks[request_id] = allocated
        return allocated
    
    def free_blocks_for_request(self, request_id: str):
        """Free all blocks allocated to a request"""
        if request_id in self.allocated_blocks:
            blocks = self.allocated_blocks.pop(request_id)
            self.free_blocks.extend(blocks)
    
    def get_free_block_count(self) -> int:
        return len(self.free_blocks)


class Scheduler:
    """
    Simplified vLLM Scheduler
    
    Implements basic scheduling logic:
    - FCFS (First Come First Served) scheduling
    - Simple preemptive scheduling based on priority
    - Block-based memory management
    """
    
    def __init__(
        self,
        num_blocks: int = 1000,
        block_size: int = 16,
        max_batch_size: int = 32
    ):
        self.block_manager = BlockManager(num_blocks, block_size)
        self.max_batch_size = max_batch_size
        self.waiting_queue: List[SchedulingRequest] = []
        self.running_requests: Dict[str, SchedulingRequest] = {}
        self.finished_requests: List[SchedulingRequest] = []
    
    def add_request(self, request: SchedulingRequest):
        """Add a new request to the scheduler"""
        request.arrival_time = time.time()
        request.status = RequestStatus.WAITING
        self.waiting_queue.append(request)
        print(f"[Scheduler] Added request {request.request_id} to waiting queue")
    
    def schedule(self) -> List[SchedulingRequest]:
        """
        Main scheduling logic
        Returns list of requests to run in this step
        """
        # Try to schedule waiting requests
        self._schedule_waiting_requests()
        
        # Get current running batch
        running_batch = list(self.running_requests.values())
        
        # Preempt if necessary
        if len(running_batch) > self.max_batch_size:
            self._preempt_low_priority(running_batch)
            running_batch = list(self.running_requests.values())
        
        return running_batch
    
    def _schedule_waiting_requests(self):
        """Try to allocate resources to waiting requests"""
        scheduled = []
        
        for request in self.waiting_queue[:]:
            # Calculate blocks needed
            total_tokens = request.prompt_length + request.max_tokens
            blocks_needed = (total_tokens + self.block_manager.block_size - 1) // self.block_manager.block_size
            
            # Try to allocate blocks
            blocks = self.block_manager.allocate_blocks(request.request_id, blocks_needed)
            if blocks is not None:
                request.allocated_blocks = len(blocks)
                request.status = RequestStatus.RUNNING
                self.waiting_queue.remove(request)
                self.running_requests[request.request_id] = request
                scheduled.append(request)
                print(f"[Scheduler] Scheduled request {request.request_id}")
            else:
                # Not enough memory, stop trying (FCFS)
                break
        
        return scheduled
    
    def _preempt_low_priority(self, running_batch: List[SchedulingRequest]):
        """Preempt lowest priority requests to free up capacity"""
        # Sort by priority (lower number = higher priority)
        sorted_requests = sorted(running_batch, key=lambda r: (r.priority, r.arrival_time))
        
        # Remove lowest priority requests until we're under max_batch_size
        while len(self.running_requests) > self.max_batch_size:
            if not sorted_requests:
                break
            request = sorted_requests.pop()
            self._preempt_request(request)
    
    def _preempt_request(self, request: SchedulingRequest):
        """Preempt a running request"""
        request.status = RequestStatus.PREEMPTED
        self.block_manager.free_blocks_for_request(request.request_id)
        del self.running_requests[request.request_id]
        # Add back to waiting queue
        request.allocated_blocks = 0
        self.waiting_queue.insert(0, request)
        print(f"[Scheduler] Preempted request {request.request_id}")
    
    def mark_finished(self, request_id: str):
        """Mark a request as finished and free its resources"""
        if request_id in self.running_requests:
            request = self.running_requests.pop(request_id)
            request.status = RequestStatus.FINISHED
            self.block_manager.free_blocks_for_request(request_id)
            self.finished_requests.append(request)
            print(f"[Scheduler] Finished request {request_id}")
    
    def get_stats(self) -> Dict:
        """Get scheduler statistics"""
        return {
            "waiting": len(self.waiting_queue),
            "running": len(self.running_requests),
            "finished": len(self.finished_requests),
            "free_blocks": self.block_manager.get_free_block_count(),
            "total_blocks": self.block_manager.num_blocks
        }


# Example usage
if __name__ == "__main__":
    # Create scheduler
    scheduler = Scheduler(num_blocks=100, block_size=16, max_batch_size=4)
    
    # Add some requests
    requests = [
        SchedulingRequest("req-1", prompt_length=50, max_tokens=100, priority=0),
        SchedulingRequest("req-2", prompt_length=30, max_tokens=150, priority=1),
        SchedulingRequest("req-3", prompt_length=100, max_tokens=200, priority=0),
        SchedulingRequest("req-4", prompt_length=20, max_tokens=50, priority=2),
        SchedulingRequest("req-5", prompt_length=80, max_tokens=120, priority=1),
    ]
    
    for req in requests:
        scheduler.add_request(req)
    
    # Run scheduling steps
    print("\n=== Scheduling Step 1 ===")
    batch = scheduler.schedule()
    print(f"Running batch: {[r.request_id for r in batch]}")
    print(f"Stats: {scheduler.get_stats()}")
    
    # Simulate finishing some requests
    print("\n=== Finishing req-1 ===")
    scheduler.mark_finished("req-1")
    print(f"Stats: {scheduler.get_stats()}")
    
    # Schedule again to fill freed slots
    print("\n=== Scheduling Step 2 ===")
    batch = scheduler.schedule()
    print(f"Running batch: {[r.request_id for r in batch]}")
    print(f"Stats: {scheduler.get_stats()}")

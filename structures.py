"""
structures.py — Custom Data Structures for Live Scraper Pipeline
================================================================
Implemented purely using Linked Nodes (no Python lists/deques for core logic).
"""

from typing import Any, Optional

class Node:
    """A generic node for linked data structures."""
    __slots__ = ("data", "next")

    def __init__(self, data: Any):
        self.data: Any = data
        self.next: Optional["Node"] = None

class PropertyQueue:
    """
    A First-In-First-Out (FIFO) queue implemented with a linked list.
    Used to hold incoming mocked properties from the live scraper.
    """
    def __init__(self):
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None
        self._size: int = 0

    def enqueue(self, item: Any) -> None:
        new_node = Node(item)
        if self.tail is None:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self._size += 1

    def dequeue(self) -> Any:
        if self.head is None:
            raise IndexError("dequeue from empty queue")
        
        popped_node = self.head
        self.head = self.head.next
        
        if self.head is None:
            self.tail = None
            
        self._size -= 1
        return popped_node.data

    def is_empty(self) -> bool:
        return self._size == 0

    def size(self) -> int:
        return self._size
        
    def peek(self) -> Any:
        if self.head is None:
            raise IndexError("peek from empty queue")
        return self.head.data

class ScraperUndoStack:
    """
    A Last-In-First-Out (LIFO) stack implemented with a linked list.
    Used to hold transaction IDs of batched inserts, allowing rollback (undo).
    """
    def __init__(self):
        self.top: Optional[Node] = None
        self._size: int = 0

    def push(self, item: Any) -> None:
        new_node = Node(item)
        new_node.next = self.top
        self.top = new_node
        self._size += 1

    def pop(self) -> Any:
        if self.top is None:
            raise IndexError("pop from empty stack")
        
        popped_node = self.top
        self.top = self.top.next
        self._size -= 1
        return popped_node.data

    def is_empty(self) -> bool:
        return self._size == 0

    def size(self) -> int:
        return self._size

    def peek(self) -> Any:
        if self.top is None:
            raise IndexError("peek from empty stack")
        return self.top.data

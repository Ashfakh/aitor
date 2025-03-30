from typing import Generic, TypeVar, Any, Optional, Callable
import asyncio
import uuid
import concurrent.futures
import threading
import logging
from aitor.aitorflows import Aitorflow
from aitor.task import Task

T = TypeVar("T")

logger = logging.getLogger(__name__)


class Aitor(Generic[T]):
    # Class-level thread pool for all aitors
    _thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=16, thread_name_prefix="AitorThread"  # Adjust based on your needs
    )

    # Class-level event loop for async operations
    _loop = asyncio.new_event_loop()

    def __init__(
        self,
        initial_memory: T,
        aitor_id: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize aitor with either new or persisted memory.
        If no aitor_id is provided, generates a new UUID.

        Args:
            initial_memory: The initial memory for this aitor
            aitor_id: Optional identifier to retrieve persisted memory
            name: Optional name for better identification
        """
        self._id: str = aitor_id or str(uuid.uuid4())
        self._name: Optional[str] = name
        self._memory: T = initial_memory
        self._workflow: Optional[Aitorflow] = None
        self._lock = threading.Lock()  # Add lock for thread-safe memory access

    @property
    def id(self) -> str:
        """
        Get the aitor's identifier.

        Returns:
            str: The aitor's unique identifier
        """
        return self._id

    @property
    def name(self) -> Optional[str]:
        """
        Get the aitor's name.

        Returns:
            Optional[str]: The aitor's name if set, None otherwise
        """
        return self._name

    @property
    def memory(self) -> T:
        with self._lock:
            return self._memory
    
    @property
    def workflow(self) -> Optional[Aitorflow]:
        """
        Get the aitor's attached workflow.
        
        Returns:
            Optional[Aitorflow]: The attached workflow, if any
        """
        return self._workflow
        
    def attach_workflow(self, workflow: Aitorflow) -> None:
        """
        Attach an Aitorflow to this aitor.
        
        Args:
            workflow: The workflow to attach
        """
        self._workflow = workflow

    def create_workflow(self, root_task: Task) -> None:
        """
        Create a new Aitorflow.
        
        Args:
            name: Optional name for the workflow
        """
        workflow = Aitorflow(name=self._name)
        workflow.add_root(root_task)
        self.attach_workflow(workflow)        
        
    def detach_workflow(self) -> None:
        """
        Remove the currently attached workflow.
        """
        self._workflow = None

    def _load_memory(self, aitor_id: str) -> T:
        """
        Load persisted memory for this aitor.
        Can be overridden by subclasses.

        Args:
            aitor_id: The identifier for the persisted memory
        Returns:
            T: The loaded memory object
        """
        pass

    async def on_receive(self, message: Any):
        """
        Handle incoming messages and orchestrate tasks.
        
        If a workflow is attached, it will be executed with the received message.

        Args:
            message: The message to be processed
        Returns:
            Any: The result of processing the message
        """
        if self._workflow:
            return await asyncio.to_thread(self._workflow.execute, message)
        return message

    async def ask(self, message: Any):
        """
        Blocking call to get a result from the aitor.

        Args:
            message: The message to be processed
        Returns:
            Any: The result from on_receive
        """
        logger.info(
            f"Aitor.ask executing - aitor_id: {self._id}, aitor_name: {self._name}, "
            f"thread_id: {threading.get_ident()}, thread_name: {threading.current_thread().name}"
        )
        return await self.on_receive(message)

    def tell(self, message: Any):
        """
        Non-blocking call to send a message to the aitor.
        Executes in a separate thread from the thread pool.

        Args:
            message: The message to be processed
        """
        def run_in_new_thread():
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                logger.info(
                    f"Aitor.tell executing - aitor_id: {self._id}, aitor_name: {self._name}, "
                    f"thread_id: {threading.get_ident()}, thread_name: {threading.current_thread().name}"
                )
                # Create and run the task
                task = loop.create_task(self.on_receive(message))
                loop.run_until_complete(task)
            except Exception as e:
                logger.error(
                    f"Error in aitor processing message - aitor_id: {self._id}, error: {str(e)}",
                    exc_info=True
                )
            finally:
                # Wait for all pending tasks to complete naturally
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.close()

        # Submit the task to thread pool
        self._thread_pool.submit(run_in_new_thread)

    def get_memory(self) -> T:
        """
        Thread-safe accessor for the aitor's local memory.

        Returns:
            T: The aitor's memory
        """
        with self._lock:
            return self._memory

    def set_memory(self, memory: T):
        """
        Thread-safe mutator for the aitor's local memory.

        Args:
            memory: The new memory object
        """
        with self._lock:
            self._memory = memory

    def persist_memory(self, external_storage: Any):
        """
        Store the aitor's memory externally.
        Can be overridden by subclasses.

        Args:
            external_storage: The storage mechanism to persist memory
        """
        pass
    
    @classmethod
    def shutdown(cls):
        """
        Cleanup method to properly shutdown the thread pool and event loop.
        Should be called when shutting down the application.
        """
        cls._thread_pool.shutdown(wait=True)
        try:
            cls._loop.stop()
            cls._loop.close()
        except:
            pass

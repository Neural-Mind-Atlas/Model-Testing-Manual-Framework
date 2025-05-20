"""Retry handler for API calls."""

import time
import logging
from typing import Callable, Any, TypeVar, Optional
from functools import wraps

T = TypeVar('T')
logger = logging.getLogger(__name__)

class RetryHandler:
    """Handles retrying failed API calls."""

    def __init__(self, max_retries: int = 3, base_delay: float = 2.0, max_delay: float = 30.0):
        """
        Initialize the retry handler.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def with_retry(self, func: Callable[..., T], *args, max_retries=None, **kwargs) -> T:
        """
        Execute a function with exponential backoff retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            max_retries: Override default max retries
            **kwargs: Keyword arguments for the function

        Returns:
            Function result

        Raises:
            Exception: If all retry attempts fail
        """
        if max_retries is None:
            max_retries = self.max_retries
            
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                # Exit early if it's the last attempt
                if attempt == max_retries:
                    break
                
                # Determine if we should retry based on error type
                should_retry = False
                
                # Check for rate limiting errors
                if any(msg in str(e).lower() for msg in [
                    "rate limit", "too many requests", "429", "quota exceeded",
                    "capacity", "usage limit", "ratelimit"
                ]):
                    should_retry = True
                    logger.warning(f"Rate limit hit on attempt {attempt+1}/{max_retries+1}, retrying...")
                
                # Check for timeout errors
                elif any(msg in str(e).lower() for msg in [
                    "timeout", "timed out", "deadline exceeded", "connection error",
                    "socket error", "network error", "server error", "service unavailable",
                    "500", "502", "503", "504"
                ]):
                    should_retry = True
                    logger.warning(f"Connection/timeout error on attempt {attempt+1}/{max_retries+1}, retrying...")
                
                # For any other potentially transient errors
                elif any(msg in str(e).lower() for msg in [
                    "temporary", "transient", "retry", "again", "unavailable",
                    "connection reset", "broken pipe", "connection closed"
                ]):
                    should_retry = True
                    logger.warning(f"Transient error on attempt {attempt+1}/{max_retries+1}, retrying...")
                    
                # Otherwise, log but still retry
                else:
                    should_retry = True  # Retry all errors by default
                    logger.warning(f"Error on attempt {attempt+1}/{max_retries+1}, retrying: {str(e)}")
                
                if should_retry:
                    # Calculate delay with exponential backoff
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    # Add small jitter to avoid thundering herd
                    jitter = delay * 0.1 * (hash(str(e)) % 10) / 10  # Deterministic jitter based on error message
                    total_delay = delay + jitter
                    logger.info(f"Retrying in {total_delay:.2f} seconds...")
                    time.sleep(total_delay)
                else:
                    # Don't retry if error isn't one we should retry
                    break

        # If we get here, all retries failed or we decided not to retry
        if last_exception:
            logger.error(f"All retry attempts failed: {str(last_exception)}")
            raise last_exception
        else:
            # This shouldn't happen, but just in case
            logger.error("All retry attempts failed with no captured exception")
            raise Exception("All retry attempts failed with no captured exception")
            
    def retry_decorator(self, max_retries=None, base_delay=None, max_delay=None):
        """
        Decorator to add retry logic to any function.
        
        Args:
            max_retries: Override default max retries
            base_delay: Override default base delay
            max_delay: Override default max delay
            
        Returns:
            Decorator function
        """
        if max_retries is None:
            max_retries = self.max_retries
        if base_delay is None:
            base_delay = self.base_delay
        if max_delay is None:
            max_delay = self.max_delay
            
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self.with_retry(func, *args, max_retries=max_retries, **kwargs)
            return wrapper
        return decorator
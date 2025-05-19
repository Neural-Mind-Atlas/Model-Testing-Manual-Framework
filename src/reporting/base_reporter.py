# src/reporting/base_reporter.py
from abc import ABC, abstractmethod
from datetime import datetime

class BaseReporter(ABC):
    """
    Base class for all reporters. Defines the common interface that all reporters must implement.
    """

    def __init__(self):
        """
        Initialize the base reporter.
        """
        self.name = "base_reporter"

    @abstractmethod
    def generate_report(self, results, output_path):
        """
        Generate a report from test results.

        Args:
            results (dict): Dictionary containing test results for one or more models
            output_path (str): Path where the report should be saved

        Returns:
            bool: True if report generation was successful, False otherwise

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement the generate_report method")
    
    def _get_current_timestamp(self):
        """
        Get the current timestamp in ISO format.
        
        Returns:
            str: Current timestamp
        """
        return datetime.now().isoformat()
    
    def _format_duration(self, seconds):
        """
        Format duration in seconds to a human-readable format.
        
        Args:
            seconds (float): Duration in seconds
            
        Returns:
            str: Formatted duration
        """
        if seconds < 1:
            return f"{seconds * 1000:.2f} ms"
        elif seconds < 60:
            return f"{seconds:.2f} seconds"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes} min {remaining_seconds:.2f} sec"
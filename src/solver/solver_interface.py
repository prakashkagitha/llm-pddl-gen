from abc import ABC, abstractmethod

class SolverInterface(ABC):
    @abstractmethod
    def solve(self, domain_file, problem_file):
        """
        Try to solve the planning problem.
        Return the plan if successful, or None if not.
        """
        pass

    @abstractmethod
    def solve_with_error(self, domain_file, problem_file):
        """
        Return a tuple (plan, error_message)
        """
        pass

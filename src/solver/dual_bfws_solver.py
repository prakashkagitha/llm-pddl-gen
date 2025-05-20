import requests
import time
from .solver_interface import SolverInterface

class DualBFWSSolver(SolverInterface):
    def __init__(self, solver="dual-bfws-ffparser"):
        self.solver = solver
        self.base_url = "https://solver.planning.domains:5001"

    def solve(self, domain_file_path, problem_file_path):
        with open(domain_file_path, "r") as f:
            domain_file = f.read()
        with open(problem_file_path, "r") as f:
            problem_file = f.read()
        attempts = 3
        for i in range(attempts):
            try:
                plan_found, result = self._send_job(domain_file, problem_file)
                if plan_found:
                    if isinstance(result, dict) and "plan" in result:
                        return result["plan"]
                    return result
            except:
                if i<attempts:
                    continue
                else:
                    raise
            break
        return None

    def solve_with_error(self, domain_file_path, problem_file_path):
        with open(domain_file_path, "r") as f:
            domain_file = f.read()
        with open(problem_file_path, "r") as f:
            problem_file = f.read()
        attempts = 3
        for i in range(attempts):
            try:
                plan_found, result = self._send_job(domain_file, problem_file)
                if plan_found:
                    if isinstance(result, dict) and "plan" in result:
                        return result["plan"], ""
                    return result, ""
                else:
                    return None, result
            except:
                if i<attempts:
                    continue
                else:
                    raise
            break
        print("Timeout **** Solver is not responding!")
        return None, "timeout"

    def _send_job(self, domain_file, problem_file):
        req_body = {"domain": domain_file, "problem": problem_file}
        solve_request_url = requests.post(f"{self.base_url}/package/{self.solver}/solve", json=req_body).json()
        celery_result = requests.post(self.base_url + solve_request_url['result'])
        while celery_result.json().get("status", "") == 'PENDING':
            celery_result = requests.post(self.base_url + solve_request_url['result'])
            time.sleep(0.5)
        result = celery_result.json()['result']
        if "Error" in celery_result.json().keys():
            return False, "timeout"
        if self.solver == "dual-bfws-ffparser":
            if result['output'] == {'plan': ''}:
                if not result['stderr']:
                    if any(kw in result['stdout'] for kw in ["NOTFOUND", "No plan", "unknown", "undeclared", "declared twice", "check input files", "does not match", "timeout"]):
                        return False, result['stdout']
                    else:
                        return True, result['stdout']
                else:
                    return False, result['stderr']
            else:
                return True, result['output']
        else:
            return True, result

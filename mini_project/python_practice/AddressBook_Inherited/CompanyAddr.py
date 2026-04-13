from Address import Addr
from dataclasses import dataclass

@dataclass
class CompanyAddr(Addr):
    company_name: str
    team_name: str

    def print_info(self):
        super().print_info()
        print(f"회사 이름: {self.company_name}")
        print(f"부서 이름: {self.team_name}")
    

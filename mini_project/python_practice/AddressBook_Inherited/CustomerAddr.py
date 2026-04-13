from Address import Addr
from dataclasses import dataclass

@dataclass
class CustomerAddr(Addr):
    company_name: str
    item: str

    def print_info(self):
        super().print_info()
        print(f"회사 이름: {self.company_name}")
        print(f"품목: {self.item}")

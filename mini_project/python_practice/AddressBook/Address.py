from dataclasses import dataclass

@dataclass
class Addr:
    name: str
    phone: str
    email: str
    address: str
    group: str = "친구"

    def print_info(self):
        print(f"이름 : {self.name}")
        print(f"전화번호 : {self.phone}")
        print(f"이메일 : {self.email}")
        print(f"주소 : {self.address}")
        print(f"그룹(친구/가족) : {self.group}")

from dataclasses import dataclass

@dataclass
class Addr:
    # 1. 타입 힌트와 함께 필드 선언(자동으로 __init__ 생성됨)
    name: str
    phone: str
    email: str
    address: str
    group: str  
    birth: str
    level: str

    # 2. 정보 출력 메서드
    def print_info(self):
        # Getter/Setter 없이 직접 속성에 접근
        print(f"이름: {self.name}")
        print(f"전화번호: {self.phone}")
        print(f"이메일: {self.email}")
        print(f"주소: {self.address}")
        print(f"그룹(회사/거래처): {self.group}")
        print(f"생일: {self.birth}")
        print(f"직급: {self.level}")

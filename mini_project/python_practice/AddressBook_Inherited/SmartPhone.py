from CompanyAddr import CompanyAddr
from CustomerAddr import CustomerAddr

class SmartPhone:

    def __init__(self):
        self.contacts = []

    def input_addr_company_data(self):
        name = input("이름을 입력하세요: ")
        phone = input("전화번호를 입력하세요: ")
        email = input("이메일을 입력하세요: ")
        address = input("주소를 입력하세요: ")
        group = "회사"
        birth = input("생일을 입력하세요 (예: 1990-01-01): ")
        level = input("직급을 입력하세요: ")
        company_name = input("회사 이름을 입력하세요: ")
        team_name = input("부서를 입력하세요: ")

        return CompanyAddr(name, phone, email, address, group, birth, level, company_name, team_name)
    
    def input_addr_customer_data(self):
        name = input("이름을 입력하세요: ")
        phone = input("전화번호를 입력하세요: ")
        email = input("이메일을 입력하세요: ")
        address = input("주소를 입력하세요: ")
        group = "거래처"
        birth = input("생일을 입력하세요 (예: 1990-01-01): ")
        level = input("직급을 입력하세요: ")
        company_name = input("회사 이름을 입력하세요: ")
        item= input("부서를 입력하세요: ")

        return CustomerAddr(name, phone, email, address, group, birth, level, company_name, item)

    def add_company_addr(self, addr):
        if len(self.contacts) < 20:
            self.contacts.append(addr)
        else:
            print("저장공간이 가득 찼습니다.")

    def add_customer_addr(self, addr):
        if len(self.contacts) < 20:
            self.contacts.append(addr)
        else:
            print("저장공간이 가득 찼습니다.")

    def print_all_addr(self):
        if not self.contacts:
            print("저장된 연락처가 없습니다.")
        else:
            for i, addr in enumerate(self.contacts):
                print(f"\n[{i+1}]")
                addr.print_info()

    def search_addr(self, name):
        for addr in self.contacts:
            if addr.name == name:
                addr.print_info()
                return
        print("등록된 연락처가 없습니다.")

    def delete_addr(self, name):
        for addr in self.contacts:
            if addr.name == name:
                self.contacts.remove(addr)
                return
        print("등록된 연락처가 없습니다.")

    def edit_addr(self, name, new_addr):
        for i, addr in enumerate(self.contacts):
            if addr.name == name:
                self.contacts[i] = new_addr
                return
        print("등록된 연락처가 없습니다.")

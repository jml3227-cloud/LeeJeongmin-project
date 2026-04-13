from Address import Addr

class SmartPhone:
    def __init__(self):
        self.contacts = []

    def input_addr_data(self):
        name = input("이름을 입력하세요")
        phone = input("전화번호를 입력하세요")
        email = input("이메일을 입력하세요")
        address = input("주소를 입력하세요")
        group = input("그룹(친구/가족)을 입력하세요")

        return Addr(name, phone, email, address, group)
    
    def add_addr(self, addr):
        if len(self.contacts) < 10:
            self.contacts.append(addr)
            print(">>> 데이터가 저장되었습니다.")
        else:
            print("연락처 저장 공간이 가득 찼습니다.")

    def print_all_addr(self):
        if not self.contacts:
            print("저장된 데이터가 없습니다.")
        else:
            for i, addr in enumerate(self.contacts):
                print(f"\n[{i+1}]")
                addr.print_info()

    def search_addr(self, name):
        for addr in self.contacts:
            if addr.name == name:
                addr.print_info()
                return
        print(f"{name}의 연락처는 찾을 수 없습니다.")

    def delete_addr(self, name):
        for addr in self.contacts:
            if addr.name == name:
                self.contacts.remove(addr)
                print(f"{name}의 연락처가 삭제되었습니다.")
                return
        print(f"{name}의 연락처는 찾을 수 없습니다.")

    def edit_addr(self, name, new_addr):
        for i, addr in enumerate(self.contacts):
            if addr.name == name:
                self.contacts[i] = new_addr
                print(f"{name}의 연락처가 수정되었습니다.")
                return
        print(f"{name}의 연락처는 찾을 수 없습니다.")


        

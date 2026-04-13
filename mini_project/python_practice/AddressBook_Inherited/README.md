# Python Address Book with Inheritance

> Advanced Contact Management System using OOP Principles
> 기존 주소록 프로그램에 상속(Inheritance) 개념을 도입하여, 회사 동료와 거래처 직원을 구분하여 관리할 수 있도록 업그레이드한 버전입니다.

## Project Overview (프로젝트 개요)
연락처 데이터의 공통된 속성(이름, 전화번호 등)은 부모 클래스(Addr)에서 관리하고, 회사(Company)와 거래처(Customer) 각각의 고유한 정보는 자식 클래스에서 관리하는 확장 가능한 구조로 설계했습니다.
또한 Python의 `@dataclass`를 활용하여 데이터 모델링의 효율성을 높였습니다.

## Tech Stack (사용 기술)
- Language: Python 3.11.9
- Core Concept: Inheritance(상속), Polymorphism(다형성), Dataclass, Module Separation

## File Structure (파일 구조)
이 프로젝트는 기능과 역할에 따라 5개의 모듈로 분리되어 있습니다.

| 파일명 | 역할 (Role) | 설명 |
|:---:|:---:|:---|
| Address.py | Parent Class | 모든 연락처의 공통 속성(이름, 번호 등)을 정의한 부모 클래스 |
| CompanyAddr.py | Child Class | 회사 동료 전용 정보를 추가한 자식 클래스 (`company_name`, `team_name`) |
| CustomerAddr.py | Child Class | 거래처 전용 정보를 추가한 자식 클래스 (`company_name`, `item`) |
| SmartPhone.py | Manager | 연락처 리스트 관리 및 비즈니스 로직 담당 |
| SmartPhoneMain.py | Main | 사용자 입력을 받고 프로그램을 실행하는 메인 파일 |

## Key Features (주요 기능)
1. 유형별 연락처 관리: '회사'와 '거래처'를 구분하여 입력받고, 서로 다른 추가 정보를 저장합니다.
2. 상속을 통한 중복 제거: 공통 기능인 `print_info()`를 부모 클래스에 정의하고, 자식 클래스에서는 `super()`를 통해 이를 재사용하며 필요한 부분만 Override했습니다.
3. 데이터 클래스 활용: `@dataclass` 데코레이터를 사용하여 생성자(`__init__`) 작성 코드를 간소화하고 가독성을 높였습니다.

## What I Learned (배운 점)

1. 상속(Inheritance)과 코드 재사용
비슷한 구조를 가진 Company와 Customer 클래스를 각각 따로 만들지 않고, Addr이라는 부모 클래스를 만들어 공통된 코드를 하나로 묶었습니다. 이를 통해 코드 중복을 줄이고 유지보수가 훨씬 쉬워진다는 점을 체감했습니다.

2. 메서드 오버라이딩 (Overriding)
부모 클래스의 print_info() 메서드를 자식 클래스에서 재정의하여 사용했습니다. 특히 super().print_info()를 통해 부모의 출력 로직을 그대로 가져오면서, 자식만의 고유한 정보만 덧붙이는 방식으로 효율적인 코드를 작성했습니다.

3. @dataclass의 효율성
단순히 데이터를 저장하는 객체를 만들 때, 매번 `__init__`과 `self.variable = variable`을 반복해서 쓰는 것이 비효율적이라고 느꼈습니다. `dataclass`를 도입함으로써 이 과정을 자동화하고, 필드 정의에만 집중할 수 있었습니다.

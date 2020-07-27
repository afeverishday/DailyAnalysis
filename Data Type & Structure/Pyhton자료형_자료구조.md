Python 자료 구조
================

모든 프로그래밍 언어가 조금씩 다른 자료 구조를 가지고 있다. 여기서는 분석에서 사용에 필요한 기본적인 Python의 자료형과
자료구조에 대해서 정리한다. Markdown 문서를 R에서 작성하고 있기 때문에 R에서 파이썬을 사용하기 위한 패키지와
사용할 파이썬 경로를 지정해서 사용함을 선언한다.

``` r
library(reticulate)
use_python("C:/Users/afeve/Anaconda3/python.exe")
```

``` python
import sys
print(sys.version)
```

    ## 3.6.6 |Anaconda custom (64-bit)| (default, Jun 28 2018, 11:27:44) [MSC v.1900 64 bit (AMD64)]

# Python 자료형

Python은 기본적으로 자료형을 선언할 필요 없이 자동으로 인식하여 입력되지만 일부 입력 데이터의 형태가 잘못 되는 경우 이를
확인하고 수정하기 위한 목적을 위해서 자료를 정리합니다.

Python이 가지는 자료형(Data Type)은 크게 숫자, 문자, 논리 등으로 크게 3가지 형태로 구분된다.(물론 이외에도
날짜를 나타내는 형태와 더 많은 자료형이 있지만 여기서는 기본적인 3가지만을 다루기로 한다.)

## 1\. 수치형(Numeric Type)

먼저 수치형은 Numeric이라고 불리며 우리가 흔히 알고 있는 숫자를 나타낸다. 수치형 또 정수 int, 실수 float,
복소수 complex 등으로 세분화 되는데 이미 중고등과정 내에서 학습된 내용이기 때문에 추가적인 설명은 생략한다.
아래 numeric\_1은 정수를, numeric\_2는 실수를, numeric\_3은 허수를 numeric\_4는 실수와
허수가 더해진 복소수를 나타낸다.

``` python
numeric_1= 1

numeric_2= 2.4

numeric_3= 2j

numeric_4= 2.4 +2j

print(numeric_1,numeric_2,numeric_3,numeric_4)
```

    ## 1 2.4 2j (2.4+2j)

## 2\. 문자형(Character Type)

문자형은 말그대로 문자를 나타내며 우리가 쓰는 언어부터 모든 특수기호와 숫자등 ’ 혹은 “로 둘러싸인 모든 자료형을 뜻한다.
여기서 ‘(apostrophe)와 “(quotation mark)의 차이는 존재하지 않는다. 문자형은 모든 자료형을
아우르는 가장 포괄적인 개념으로 어떤 자료형이라도 ’(apostrophe)나”(quotation mark)로 나타내어
진다면 문자로 인식한다. 따라서 1은 숫자형이지만 ’1’ 혹은”1"은 문자형이다. 유의할 사항은 대소문자 구별을 확실히
해야한다는 것이다.

``` python
character_1= 'R'

character_2= '1'

character_3= '&'
  
print(character_1,character_2,character_3)
```

    ## R 1 &

## 3.논리형(Logical Type)

논리형은 참, 거짓이라는 컴퓨터에 가장 기본이 되는 자료형으로 True, False 두가지 값만을 가진다. 언어별로 대소문자가
조금씩 다르지만 python에서는 크게 True, False와 같이 앞글자만 대문자로 쓴다.

``` python

logical_1= True

logical_2= False

print(logical_1,logical_2)
```

    ## True False

## 4\. Date 자료형

Date 자료형 날짜 자료형은 날짜를 표현하기 위해서 사용되는 형태로 형식을 지정하여 사용할수 있습니다. 기본적으로 Pyhon에서
제공하는 날짜 자료형은 존재하지 않지만 날짜 자료형은 데이터분석에 있어 중요한 자료형이기 때문에 이를 사용할수 있는 패키지를 통해
날짜 자료형을 다루어 본다. 날짜 자료형을 다루는 패키지는 datetime 패키지이며 내부에 날짜와 시간을 다루는
datetime, 날짜만 다루는 date, 시간만 다루는 time, 시간 구간 정보를 저장하는 timedelta클래스 등이 있다.

``` python
import datetime

date_0 =datetime.datetime.now()

date_1 = datetime.datetime.strptime("2020-07-13 01:00", "%Y-%m-%d %H:%M")

date_2 = date_0- date_1

print(date_0,date_1,date_2)
```

    ## 2020-07-14 16:08:56.973938 2020-07-13 01:00:00 1 day, 15:08:56.973938

``` python
print(date_0.year,date_0.month, date_0.weekday(),date_0.day)
```

    ## 2020 7 1 14

``` python
print( date_0.hour, date_0.minute, date_0.second, date_0.microsecond)
```

    ## 16 8 56 973938

ㄴ

\#\#5. NA, NULL 자료형

보통 파이썬에서는 문자열이나 빈 데이터는 별도의 범주형 데이터를 사용하며, 실수형, 정수형, 날짜형 변수에 결측치 처리가
사용된다. Python은 결측값의 표현 방식을 NaN, None, np.nan의 세 가지 형태로 나타내며 각각의
결측값의 타입이 다르게 표현된다. 먼저 None은 Nonetype이나 np.nan과 Na는 flaot 타입으로 인식하고
결측값은 NaN으로 표시된다. 다만, 이때 주의할 사항은 정수형의 경우 결측값을 나타내는 NaN이 없기 때문에 결측값이
있는 경우 자동으로 실수형으로 변환된다. 또, 날짜형 변수도 날짜 시간형의 경우에만 결측값을 인식하고 이때 결측값은
NaT로 표현된다. 추가적으로 무한대 값은 float(‘inf’) 형태로 나타낼수 있다.

``` python
import numpy as np

etc_1 = None 
etc_2 = np.nan
etc_3 = float('nan')
etc_4 = float('inf')     # positive infinity
etc_5 =  float('-inf')    # negative infinity

print(etc_1,etc_2,etc_3,etc_4,etc_5)
```

    ## None nan nan inf -inf

위 자료형에 대한 설명이 끝났다면 제대로 자료형이 지정되었는지 이를 확인하는 과정도 필요하다. 그때 사용되는 함수는 type()
함수로 확인할 수 있다. 사용법은 간단하다. 아래 결과를 통해서 알수 있지만 숫자형, 문자형, 논리형의 경우 각각
int,float,complex, str, bool의 정수,실수,복소수, 문자, 논리 자료형을 반환하지만 None의 경우
NoneType을 나머지 np.nan과 무한대 등은 실수형을 반환한다.

``` python
type(numeric_1)
```

    ## <class 'int'>

``` python
type(numeric_2)
```

    ## <class 'float'>

``` python
type(numeric_3)
```

    ## <class 'complex'>

``` python
type(character_1)
```

    ## <class 'str'>

``` python
type(logical_1)
```

    ## <class 'bool'>

``` python
type(etc_1)
```

    ## <class 'NoneType'>

``` python
type(etc_2)
```

    ## <class 'float'>

``` python
type(etc_3)
```

    ## <class 'float'>

``` python
type(etc_5)
```

    ## <class 'float'>

또 각 자료형이 잘못 입력되었거나 목적에 맞지 않다면 서로 간에 변형이 가능한데 이때 사용되는 함수는 int(), float(),
str(), bool() 등의 함수를 사용한다. 이때 주의할 사항은 변수간에 변형이 가능한지 확인하는 것이다.

모든 자료형은 문자형으로 변환가능하다. 문자가 숫자로 이루어졌다면 숫자형 변수로 변환가능하다. 숫자형 변수와 논리형 변수는 서로
변환 가능하다.(0은 False, 나머지 True)

``` python
float(numeric_1)
```

    ## 1.0

``` python
int(numeric_2)
```

    ## 2

``` python
str(numeric_2)
```

    ## '2.4'

``` python
bool(0)
```

    ## False

``` python
bool(character_1)
```

    ## True

# Python 자료 구조

위에서 살펴본 자료형 데이터를 기본으로 Python의 자료 구조는 리스트, 튜플, 딕셔너리, 셋 총 네 가지의 형태를 가진다.

## 0\. 스칼라

먼저 스칼라는 위에서 살펴본 자료형 데이터 하나 하나를 뜻한다.

## 1\. 리스트

리스트는 python에서 사용하는 자료 구조 중 가장 기본이 되는 스칼라가 나열된 형태로 R과 다른 점은 한 리스트트 내에 다른
데이터 타입을 모두 담을수 있다는 것이다. 예를 들어 문자형과 숫자형, 논리형을 나란히 연결하여 만들어도 문제가 되지 안흔다.
사용법은 대괄호로 감싼 뒤\[\]값을 ,로 나열하면 된다. 리스트의 특징은 리스트 안에 또 다른 리스트를 담을 수 있다는 것이며,
권장하지는 않는다.

``` python
list_1= [1,2,3,4,5]
list_2= ['one', 'Two', 'Three', 'Four', 'Five']
list_3= [True, False, True, False, True]
list_4= [1,2,3,'Four', True]

list_5=[[list_1],[list_3]]

print(list_1,list_2, list_3, list_4)
```

    ## [1, 2, 3, 4, 5] ['one', 'Two', 'Three', 'Four', 'Five'] [True, False, True, False, True] [1, 2, 3, 'Four', True]

``` python
print(list_5)
```

    ## [[[1, 2, 3, 4, 5]], [[True, False, True, False, True]]]

각 리스트에서 특정 성분을 불러오기 위해서는 인덱스를 사용해야 한다. R과 달리 인덱스는 0부터 시작된다

``` python
list_1[1]
```

    ## 2

``` python
list_2[2:3]
```

    ## ['Three']

``` python
list_3[::-1]
```

    ## [True, False, True, False, True]

``` python
list_4[-1:]
```

    ## [True]

``` python
list_5[1]
```

    ## [[True, False, True, False, True]]

## 2\. 튜플: 데이터 변경 불가능

리스트와 기능적으로 거의 동일하지만, 수정이 불가능한 특성이 있음. 따라서 리스트에서 사용하던 append, pop, remove
같은 기능을 지원하지 않음. 중요한 데이터, 변형이 불가능한 데이터에 할당됨. 사용법은 소괄호로 감싼 뒤()값을 ,로 나열하면
된다. 튜플도 여러가지 데이터 타입을 담을 수 있고, 중첩이 가능하다.

``` python
tuple_1= (1,2,3,4,5)
tuple_2= ('one', 'Two', 'Three', 'Four', 'Five')
tuple_3= (True, False, True, False, True)
tuple_4= (1,2,3,'Four', True)

tuple_5=((tuple_1),(tuple_3))

print(tuple_1,tuple_2, tuple_3, tuple_4)
```

    ## (1, 2, 3, 4, 5) ('one', 'Two', 'Three', 'Four', 'Five') (True, False, True, False, True) (1, 2, 3, 'Four', True)

``` python
print(tuple_5)
```

    ## ((1, 2, 3, 4, 5), (True, False, True, False, True))

각 튜플에서 특정 성분을 불러오기 위한 인덱싱은 리스트와 동일하다.

``` python
tuple_1[1]
```

    ## 2

``` python
tuple_2[2:3]
```

    ## ('Three',)

``` python
tuple_3[::-1]
```

    ## (True, False, True, False, True)

``` python
tuple_4[-1:]
```

    ## (True,)

``` python
tuple_5[1]
```

    ## (True, False, True, False, True)

## 3\. 집합(set): 중복을 허용하지 않는 데이터 집합

집합 기능을 제공하는 자료구조, 리스트나 튜플의 중복을 제거하기 위해서 사용됩니다. 다만 순서를 무시함. 사용법은 중괄호를 {}
사용함.

``` python
set_1= {1,2,3,4,5,6,7,7,7,7,7,7}
set_2={1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3}
set_3=('one', 'one', 'two', 'two')

print(set_1, set_2, set_3)
```

    ## {1, 2, 3, 4, 5, 6, 7} {1, 2, 3} ('one', 'one', 'two', 'two')

집합자료는 인덱싱이 아니라 주로 집합의 기호를 연산으로 사용하기 위해서 쓰이기 때문에 차집합, 합집합, 교집합, 여집합에 대해서
살펴 보았다.

``` python
cha= set_1-set_2
hap= set_1|set_2
gop= set_1&set_2
yeo= set_1^set_2

print(cha, hap, gop, yeo)
```

    ## {4, 5, 6, 7} {1, 2, 3, 4, 5, 6, 7} {1, 2, 3} {4, 5, 6, 7}

## 4\. 딕셔너리(Dict): key & value

딕셔너리는 키와 value이 쌍으로 구성된 집합으로 key는 고유값으로 중복을 허용하지 않는다. 집합이기때문에 set과 동일하게
중괄호{}를 사용한다. 인덱싱을 지원하지 않으며 인덱싱 대신 키 값을 사용한다. 또, 리스트나 튜플과 다르게 데이터를 순서대로
저장하지 않는다는 특징이 있다.

``` python
dict_1 = {'English':90, 'Math':99, 'Kor':89}  

dict_2 = {'speaking':90, 'listening':99, 'writing':89} 

print(dict_1, dict_2)
```

    ## {'English': 90, 'Math': 99, 'Kor': 89} {'speaking': 90, 'listening': 99, 'writing': 89}

어레이에서 특정 성분을 불러오기 위한 인덱싱은 행렬에서 하나의 인덱싱(깊이)이 추가된 형태입니다.

``` python

print(dict_1.keys())
```

    ## dict_keys(['English', 'Math', 'Kor'])

``` python
print(dict_1.values())
```

    ## dict_values([90, 99, 89])

``` python
print(dict_1.items())
```

    ## dict_items([('English', 90), ('Math', 99), ('Kor', 89)])

``` python
print('English' in dict_1)
```

    ## True

``` python
print(dict_1['English'])
```

    ## 90

앞의 내용은 Native Python 내 자료 구조이고, 아래 내용은 데이터 분석시 필요한 numpy와 pandas에서 사용하는
자료 구조이다.

## 5\. array: 다차원 배열

numpy가 제공하는 자료구조는 다차원 배열인 array를 지원한다. array는 1차원부터 3차원까지의 array를 사용하며,
배열의 구조는 shape으로 표시한다. 이때 제공되는 shape 배열의 구조는 파이썬 튜플 자료형으로 정의된다.

``` python

array_1 = np.array([1, 2, 3])

array_2 = np.array([(1,2,3), (4,5,6)], dtype = float)

array_3 = np.array([[[1,2,3], [4,5,6]], [[3,2,1], [4,5,6]]], dtype = float)

print( array_1, array_2, array_3)
```

    ## [1 2 3] [[1. 2. 3.]
    ##  [4. 5. 6.]] [[[1. 2. 3.]
    ##   [4. 5. 6.]]
    ## 
    ##  [[3. 2. 1.]
    ##   [4. 5. 6.]]]

``` python

print(array_1.shape, array_2.shape, array_3.shape)
```

    ## (3,) (2, 3) (2, 2, 3)

pandas는 panel dataset의 약자로 python에서 R처럼 데이터 분석을 하기 위해 제공하는 강력한 분석용
패키지이다. 판다스에서 제공하는 자료 구조는 크게 1차원 Series, 2차원 DataFrame, 3차원
Panel 등이다.

## 6\. Series: 1차원 자료구조

Python pandas Series는 R에서 벡터 형태의 자료 구조와 거의 동일한 형태라고 생각할수 있다. 리스트, 튜플,
딕셔너리 모두 시리즈 데이터로 변형가능하다. 라벨을 지정하지 않으면 0부터 시작하는 값이 할당되며, 딕셔너리는
key값을 라벨로 자동할당된다.

``` python
import pandas as pd

series_1 = pd.Series(list_1)
series_2 = pd.Series(tuple_1)
series_3 = pd.Series(dict_1)

print(series_1, series_2, series_3)
```

    ## 0    1
    ## 1    2
    ## 2    3
    ## 3    4
    ## 4    5
    ## dtype: int64 0    1
    ## 1    2
    ## 2    3
    ## 3    4
    ## 4    5
    ## dtype: int64 English    90
    ## Math       99
    ## Kor        89
    ## dtype: int64

## 7\. DataFrame: 2차원 자료구조

Python pandas DataFrame은 R에 데이터프레임과 자료 구조와 거의 동일한 형태라고 생각할수 있다. 엑셀시트와 같은
형태라고 생각하면 되며, 행과 열로 이루어진 2차원 데이터이다. 보통 행의 경우는 인덱싱 된 숫자로, 열의 경우는 열의 이름을
통해 인덱싱한다.

``` python
import pandas as pd

dataframe_1= pd.DataFrame({'A':list_1, 'B':tuple_2})

print(dataframe_1)
```

    ##    A      B
    ## 0  1    one
    ## 1  2    Two
    ## 2  3  Three
    ## 3  4   Four
    ## 4  5   Five

## 8\. Panel: 3차원 자료구조

Python pandas Panel은 R에 리스트 자료 구조와 거의 동일한 형태라고 생각할수 있다. 여러 개의 자료구조가 합쳐져
있는 경우라고 생각하면 되는데, 3차원이기 때문에 3개의 축이 존재하게 된다. R에서 이와 비슷한 인덱싱은 리스트가 아니라
어레이에서 사용되는데 이때 인덱싱은 2,2,2의 형태로 앞에서 부터 2행 2열의 2번째 행렬이라고 읽지만 판다스
panel의 경우는 첫번째 인덱싱이 아이템 자료구조, 그 다음 인덱싱이 해당 자료 구조 중에서 행, 그다음이 열 값을
나타낸다. 아쉽게도 panel 자료구조는 pandas 0.25 이후에 deprecated 되어서 사용할 수
없게되었다.

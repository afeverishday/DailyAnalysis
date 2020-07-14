Data Analysis Tutorial
================

## Introduce

This is an R Markdown format used for publishing markdown documents to
GitHub.

이 문서는 markdown 문서를 github로 publishing 하기위해서 사용되는 R markdown 형식이다.

-----

### 단축키

가) chunk생성: Ctrl + Alt + I

나) knit하기: Ctrl + Shift + k

다) 한줄 실행: Ctrl + Enter, chunk실행: Ctrl + Shift + Enter

### 텍스트 작성방법

가) \#: \#을 앞에 붙이면 굵은 문자 스타일로 Viewer창에 출력된다. 많이 붙일수록 더 굵은 문자로 표기된다.

나) —: -기호를 세 번 입력하면 줄이 표시된다.

다) \*: 글머리 기호가 된다.

### 표 작성 방법

| 이름 | 컬럼 |
| -- | -- |
| K  | JH |
| K  | EY |

## 코드내장하기

가) r 코드를 백틱(\`)으로 감싼다. R이 인라인 코드를 실행된 결과로 대체한다

2 더하기 2는 4와 같다

나) R코드 덩어리를 `{r} 으로 시작하고.` 으로 마무리한다

    ## [1] 150   5

## 화면 출력 선택 옵션

knitr 선택옵션을 사용해서 코드 덩어리 출력 스타일을 적용한다. 코드 상단 괄호 내부에 선택옵션을 지정한다 Here’s
some code

가) 코드를 보여줌

나) 결과를 보여줌

    ## [1] 150   5

전체 옵션은 knitr::opts\_chunk$get 함수로 확인할 수 있다.

### table

1.  직접 입력하기

| 선택옵션       | 기본설정                                      | 효과                              |
| ---------- | ----------------------------------------- | ------------------------------- |
| eval       | TRUE                                      | 코드를 평가하고 실행결과를 포함한다.            |
| echo       | TRUE                                      | 실행결과와 함께 코드를 출력한다.              |
| warning    | TRUE                                      | 경고메시지를 출력한다.                    |
| error      | FALSE                                     | 오류메시지를 출력한다.                    |
| message    | TRUE                                      | 메시지를 출력한다.                      |
| tidy       | FALSE                                     | 깔끔한 방식으로 코드 형태를 변형한다.           |
| results    | “markup” “markup”, “asis”, “hold”, “hide” |                                 |
| cache      | FALSE                                     | 결과값을 캐쉬해서 향후 실행시 건너뛰게 설정한다.     |
| comment    | “\#\#”                                    | 주석문자로 출력결과에 서두를 붙인다.            |
| fig.width  | 7                                         | 덩어리로 생성되는 그래프에 대한 폭을 인치로 지정한다.  |
| fig.height | 7                                         | 덩어리로 생성되는 그래프에 대한 높이을 인치로 지정한다. |

2.  kable을 활용하기

| Sepal.Length | Sepal.Width | Petal.Length | Petal.Width | Species |
| -----------: | ----------: | -----------: | ----------: | :------ |
|          5.1 |         3.5 |          1.4 |         0.2 | setosa  |
|          4.9 |         3.0 |          1.4 |         0.2 | setosa  |
|          4.7 |         3.2 |          1.3 |         0.2 | setosa  |
|          4.6 |         3.1 |          1.5 |         0.2 | setosa  |
|          5.0 |         3.6 |          1.4 |         0.2 | setosa  |

A caption

## Plots 내장하기

<img src="Markdown_index_files/figure-gfm/pressure-1.jpeg" style="display: block; margin: auto;" />

<div class="figure" style="text-align: center">

<img src="https://www.tidyverse.org/images/tidyverse-default.png" alt="tidyverse logo"  />

<p class="caption">

tidyverse logo

</p>

</div>

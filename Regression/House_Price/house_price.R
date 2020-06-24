# house price

library(data.table)
library(dplyr)
train<-fread("Regression/House_Price/train.csv", stringsAsFactors = F)
test<-fread("Regression/House_Price/test.csv", stringsAsFactors = F)

# ID : 집을 구분하는 번호
# date : 집을 구매한 날짜
# price : 집의 가격(Target variable)
# bedrooms : 침실의 수
# bathrooms : 화장실의 수
# sqft_living : 주거 공간의 평방 피트(면적)
# sqft_lot : 부지의 평방 피트(면적)
# floors : 집의 층 수
# waterfront : 집의 전방에 강이 흐르는지 유무 (a.k.a. 리버뷰)
# view : 집이 얼마나 좋아 보이는지의 정도
# condition : 집의 전반적인 상태
# grade : King County grading 시스템 기준으로 매긴 집의 등급
# sqft_above : 지하실을 제외한 평방 피트(면적)
# sqft_basement : 지하실의 평방 피트(면적)
# yr_built : 지어진 년도
# yr_renovated : 집을 재건축한 년도
# zipcode : 우편번호
# lat : 위도
# long : 경도
# sqft_living15 : 2015년 기준 주거 공간의 평방 피트(면적, 집을 재건축했다면, 변화가 있을 수 있음)
# sqft_lot15 : 2015년 기준 부지의 평방 피트(면적, 집을 재건축했다면, 변화가 있을 수 있음)


# EDA
dim(train)  # data 구조 확인
str(train) # 변수 확인
summary(train)
head(train)
tail(train)

test_labels <- test$id  # 후에 submission을 위해 test의 id는 벡터에 두고, 변수는 삭제한다.
test$id <- NULL
train$id <- NULL
test$price <- NA     # rbind 사전 작업으로 변수 개수 맞추기 위해 SalePrice 변수 생성
train$source <- "train"
test$source <-"test"
all <- rbind(train, test)

dim(all)
str(all) # 변수 확인
head(all)
glimpse(all)



library(ggplot2)
library(scales)
options(scipen = 10)
ggplot(data = all[!is.na(all$price),], aes(x = price)) +
  geom_histogram(fill = 'blue', binwidth = 100000) +
  scale_x_continuous( labels = comma) 
#0~80만까지 10만 단위로 x축 표현(구분자 ,) breaks = seq(0, 800000, by = 10000),

all<-as.data.frame(all)
summary(all)


#문자대체
all$date<-gsub("T","", all$date)
all$date
all$date<-as.POSIXct(all$date, format="%Y%m%d%H%M", origin="1970-01-01")

colnames(all)



colSums(is.na(all))
# 숫자 변수중 factor로 의심되는 변수는 waterfront, yr_built, zipcode, yr_renovated정도이다.

#library(dummies)
#d<-dummy.data.frame(all)
all$waterfront<-as.factor(all$waterfront)


all$renovate_yn= ifelse(all$yr_renovated>0, 1,0)
all$age <- ifelse(all$renovate_yn == 0 , year(all$date) -all$yr_built, year(all$date) - all$yr_renovated)

table(all$zipcode)


numericVars <- which(sapply(all, is.numeric)) # index 벡터 numeric 변수 저장
numericVarNames <- names(numericVars) #이름 명명하여 변수 생성
cat('There are', length(numericVars), 'numeric variables')

characterVars <- which(sapply(all, is.character)) # index 벡터 character 변수 저장
characterVarNames <- names(characterVars) #이름 명명하여 변수 생성
cat('There are', length(characterVars), 'character variables')

factorVars <- which(sapply(all, is.factor)) # index 벡터 factor 변수 저장
factorVarNames <- names(factorVars) #이름 명명하여 변수 생성
cat('There are', length(factorVars), 'factor variables')

dateVars <- 1 # index 벡터 character 변수 저장
dateVarNames <- names(dateVars) #이름 명명하여 변수 생성
cat('There are', length(dateVars), 'date variables')

all %>% select(i) %>% unique() %>% sort(decreasing = T)




all_numVar <- all[, numericVars]
cor_numVar <- cor(all_numVar, use='pairwise.complete.obs') # 전 numeric 변수의 상관 계수

# SalePrice와의 상관 계수 내림차순으로 정렬
cor_sorted <- as.matrix(sort(cor_numVar[, 'price'], decreasing = TRUE))
# 상관 계수가 큰 변수만을 선택
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x) > 0.5)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]

library(corrplot)
corrplot.mixed(cor_numVar, 
               tl.col = 'black',   # 변수명 색깔
               tl.pos = 'lt',      # 변수명 왼쪽 표시
               number.cex = .7)    # matrix안 상관계수 text 크기

glimpse(all)
library(dplyr)
all %>% group_by(waterfront) %>% summarise(mean=mean(price/sqft_living, na.rm = TRUE))
all %>% group_by(zipcode) %>% summarise(mean=mean(price/sqft_living, na.rm = TRUE)) 
all %>% group_by() %>% summarise(mean=mean(price/sqft_living, na.rm = TRUE)) 



## 수치형 변수 표준화
X_train_num <- all[filter=="train",num_vars, with=F]
X_test_num <- all[filter=="test",num_vars, with=F]

mean.tr <- apply(X_train_num, 2, mean)
sd.tr <- apply(X_train_num, 2, sd)

X_train_num <- scale(X_train_num, center=mean.tr, scale=sd.tr)
X_test_num <- scale(X_test_num, center=mean.tr, scale=sd.tr)

X_train <- model.matrix(~.-1, data=cbind(X_train_num, full[filter=="train", cat_vars, with=F])) 
X_test <- model.matrix(~.-1, data=cbind(X_test_num, full[filter=="test", cat_vars, with=F]))
Y_train <- log(full[filter=="train", price])











ggplot(data = all[!is.na(all$price),], aes(x = factor(grade), y = price)) +
  geom_boxplot(col = 'blue') + labs(x = 'Overall Quality') +
  scale_y_continuous( labels = comma) 
#0~80만까지 10만단위로 y축 표현(구분자 ,)

library(ggrepel)
ggplot(data = all[!is.na(all$price),], aes(x = sqft_above , y = price)) +
  geom_point(col = 'blue') + 
  geom_smooth(method = 'lm', se = FALSE, color = 'black', aes(group = 1)) +
  scale_y_continuous( labels = comma) +
  geom_text_repel(aes(label = ifelse(all$sqft_above[!is.na(all$price)] > 500, #price 7500이상 텍스트 표기
                                     rownames(all), '')))

all[c(2776,5109,6470,8913,13810), c('price', 'sqft_above', 'grade')]


NAcol <- which(colSums(is.na(all)) > 0)  # 모든 결측치 변수 생성
sort(colSums(sapply(all[NAcol], is.na)), decreasing = TRUE) #결측치 변수 별로 내림차순 정렬

cat('There are', length(NAcol), 'columns with  missing values')

# house price

library(data.table)
library(dplyr)
train<-fread("Regression/House_Price_raw/train.csv", stringsAsFactors = F)
test<-fread("Regression/House_Price_raw/test.csv", stringsAsFactors = F)

# EDA
dim(train)  # data 구조 확인
dim(test)
str(train) # 변수 확인
str(test)
head(train)
head(test)
tail(train)
tail(test)
     
test_labels <- test$Id  # 후에 submission을 위해 test의 id는 벡터에 두고, 변수는 삭제한다.
test$Id <- NULL
train$Id <- NULL

test$SalePrice <- NA     # rbind 사전 작업으로 변수 개수 맞추기 위해 SalePrice 변수 생성
all <- rbind(train, test)


dim(all)
str(all) # 변수 확인
head(all)


library(ggplot2)
library(scales)
options(scipen = 10) # 모든 숫자표현
ggplot(data = all[!is.na(all$SalePrice),], aes(x = SalePrice)) +
  geom_histogram(fill = 'blue', binwidth = 10000) +
  scale_x_continuous( labels = comma) 
#0~80만까지 10만 단위로 x축 표현(구분자 ,) breaks = seq(0, 800000, by = 10000),

all<-as.data.frame(all)
summary(all)



numericVars <- which(sapply(all, is.numeric)) # index 벡터 numeric 변수 저장
numericVarNames <- names(numericVars) #이름 명명하여 변수 생성
cat('There are', length(numericVars), 'numeric variables')

all_numVar <- all[, numericVars]
cor_numVar <- cor(all_numVar, use='pairwise.complete.obs') # 전 numeric 변수의 상관 계수

# SalePrice와의 상관 계수 내림차순으로 정렬
cor_sorted <- as.matrix(sort(cor_numVar[, 'SalePrice'], decreasing = TRUE))
# 상관 계수가 큰 변수만을 선택
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x) > 0.5)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]

library(corrplot)
corrplot.mixed(cor_numVar, 
               tl.col = 'black',   # 변수명 색깔
               tl.pos = 'lt',      # 변수명 왼쪽 표시
               number.cex = .7)    # matrix안 상관계수 text 크기


ggplot(data = all[!is.na(all$SalePrice),], aes(x = factor(OverallQual), y = SalePrice)) +
  geom_boxplot(col = 'blue') + labs(x = 'Overall Quality') +
  scale_y_continuous( labels = comma) 
#0~80만까지 10만단위로 y축 표현(구분자 ,)

library(ggrepel)
ggplot(data = all[!is.na(all$SalePrice),], aes(x = GrLivArea , y = SalePrice)) +
  geom_point(col = 'blue') + 
  geom_smooth(method = 'lm', se = FALSE, color = 'black', aes(group = 1)) +
  scale_y_continuous( labels = comma) +
  geom_text_repel(aes(label = ifelse(all$GrLivArea[!is.na(all$SalePrice)] > 4500, #price 7500이상 텍스트 표기
                                     rownames(all), '')))

# 이상치에 해당하는 데이터의 가격과 품질 확인
out_rowname<-which(all$GrLivArea[!is.na(all$SalePrice)] > 4500)
all[out_rowname, c('SalePrice', 'GrLivArea', 'OverallQual')]

#결측치 확인
NAcol <- which(colSums(is.na(all)) > 0)  # 모든 결측치 변수 생성
sort(colSums(sapply(all[NAcol], is.na)), decreasing = TRUE) #결측치 변수 별로 내림차순 정렬

cat('There are', length(NAcol), 'columns with  missing values')

# Pool 변수
table(all$PoolQC)

all$PoolQC<-ifelse(is.na(all$PoolQC),'None',all$PoolQC)
Qualities <- c('None' = 0, 'Po' = 1, 'Fa' = 2, 'TA' = 3, 'Gd' = 4, 'Ex' = 5)
library(plyr)
all$PoolQC <- as.integer(revalue(all$PoolQC, Qualities))

table(all$PoolArea)
all %>% filter(PoolQC==0 & PoolArea>0) %>% select(PoolArea, PoolQC, OverallQual)
all$PoolQC<- ifelse(all$PoolQC==0 & all$PoolArea>0, all$OverallQual, all$PoolQC)


# miscellanuous 잡다한 
table(all$MiscFeature)
all$MiscFeature<-ifelse(is.na(all$MiscFeature),'None',all$MiscFeature)

all$MiscFeature <- as.factor(all$MiscFeature)
ggplot(all[complete.cases(all$SalePrice),] , aes(x = MiscFeature, y = SalePrice)) +
  geom_bar(stat = 'summary', fun.y = 'median', fill = 'blue') +
  scale_y_continuous(breaks = seq(0, 800000, by = 100000), labels = comma) +
  geom_label(stat = 'count', aes(label = ..count.., y = ..count..)) #막대 그래프 count 라벨링

# Alley 샛길,뒷길
table(all$Alley)
all$Alley<-ifelse(is.na(all$Alley),'None',all$Alley)
all$Alley<- as.factor(all$Alley)

ggplot(all[!is.na(all$SalePrice),], aes(x = Alley, y = SalePrice)) + 
  geom_bar(stat = 'summary', fun.y = 'median', fill = 'blue') +
  scale_y_continuous(breaks = seq(0, 200000, by = 50000), labels = comma)

# Fence
table(all$Fence)
all$Fence <- ifelse(is.na(all$Fence),'None', all$Fence)

all[!is.na(all$SalePrice),] %>% 
  group_by(Fence) %>%  #Fence 그룹핑
  summarise(median = median(SalePrice), counts = n()) #Fence변수의 price 중위값, 개수 확인
# => no fence is best

all$Fence <- as.factor(all$Fence)

# Fireplace quality
table(all$Fireplaces)
sum(table(all$Fireplaces))
table(all$FireplaceQu)
all$FireplaceQu <- ifelse(is.na(all$FireplaceQu),'None',all$FireplaceQu)
#FireplaceQu 결측치의 수는 fireplaces 변수가 0인 수와 일치한다.
all$FireplaceQu<-revalue(all$FireplaceQu, Qualities) %>% as.integer()

# Lot variables
# LotFrontage: Linear feet of street connected to property
# LotShape: General shape of property
# LotConfig: Lot configuration
table(all$LotFrontage)
sum(table(all$LotFrontage))

ggplot(all[!is.na(all$LotFrontage),], 
       aes(x = as.factor(Neighborhood), y = LotFrontage)) +
  geom_bar(stat = 'summary', fun.y = 'median', fill = 'blue') + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Text 45도 기울기, 높이는 1로 설정

for (i in 1:nrow(all)) {
  if(is.na(all$LotFrontage[i])){
    all$LotFrontage[i] <- as.integer(median(all$LotFrontage[all$Neighborhood==all$Neighborhood[i]],
                                            na.rm = TRUE))
  }
}

table(all$LotShape)
sum(table(all$LotShape))
all$LotShape <- as.integer(revalue(all$LotShape, c('IR3' = 0, 'IR2' = 1, 'IR1' = 2, 'Reg' = 3)))


table(all$LotConfig)
sum(table(all$LotConfig))
ggplot(all[!is.na(all$SalePrice),], aes(x = as.factor(LotConfig), y = SalePrice)) +
  geom_bar(stat = 'summary', fun.y = 'median', fill = 'blue') +
  scale_y_continuous(breaks = seq(0, 800000, by = 100000), labels = comma) +
  geom_label(stat = 'count', aes(label = ..count.., y = ..count..))

all$LotConfig <- as.factor(all$LotConfig)

#  Garage 관련
table(all$GarageYrBlt)
sum(table(all$GarageYrBlt))
all$GarageYrBlt<- ifelse(is.na(all$GarageYrBlt), all$YearBuilt, all$GarageYrBlt)


table(all$GarageType)
sum(table(all$GarageType))

table(all$GarageFinish)
sum(table(all$GarageFinish))

table(all$GarageCond)
sum(table(all$GarageCond))

table(all$GarageQual)
sum(table(all$GarageQual))

#157개의 결측치가 159개 결측치의 변수와 동일한 관측치인지 확인해 보겠다.
length(which(is.na(all$GarageType) & is.na(all$GarageFinish) & is.na(all$GarageCond) & is.na(all$GarageQual)))


library(knitr)
#나머지 2개의 관측치를 찾아보겠다.
kable(all[!is.na(all$GarageType) & is.na(all$GarageFinish), 
          c('GarageCars', 'GarageArea', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish')])

# 최빈도 값으로 결측치 대체
all$GarageCond[2127] <- names(sort(-table(all$GarageCond)))[1]
all$GarageQual[2127] <- names(sort(-table(all$GarageQual)))[1]
all$GarageFinish[2127] <- names(sort(-table(all$GarageFinish)))[1]

# 대체 후 값 확인
kable(all[2127, c('GarageYrBlt', 'GarageCars', 'GarageArea', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish')])

# 2577 house에 값을 할당
all$GarageCars[2577] <- 0
all$GarageArea[2577] <- 0
all$GarageType[2577] <- 'None'

# 문자형 변수인 4 변수의 결측치가 모두 158개인지 확인해보겠다.
length(which(is.na(all$GarageType) & is.na(all$GarageFinish) & is.na(all$GarageCond) & is.na(all$GarageQual)))


all$GarageType[is.na(all$GarageType)] <- 'No Garage'
all$GarageType <- as.factor(all$GarageType)
table(all$GarageType)

all$GarageFinish[is.na(all$GarageFinish)] <- 'None'
Finish <- c('None' = 0, 'Unf' = 1, 'RFn' = 2, 'Fin' = 3) #문자형 수치형으로 변환

all$GarageFinish <- as.integer(revalue(all$GarageFinish, Finish))
table(all$GarageFinish)

all$GarageQual[is.na(all$GarageQual)] <- 'None'
all$GarageQual <- as.integer(revalue(all$GarageQual, Qualities))
table(all$GarageQual)

all$GarageCond[is.na(all$GarageCond)] <- 'None'
all$GarageCond <- as.integer(revalue(all$GarageCond, Qualities))
table(all$GarageCond)


# Basement 관련 변수
# 79개의 결측치가 80 이상의 결측치의 값과 동일한 관측치를 보이는지 확인하겠다.
length(which(is.na(all$BsmtQual) & is.na(all$BsmtCond) & is.na(all$BsmtExposure) & 
               is.na(all$BsmtFinType1) & is.na(all$BsmtFinType2)))

# 추가 결측치 찾기: BsmtFinType1은 결측치가 아니지만, 다른 4개 변수들이 결측치인 경우
all[!is.na(all$BsmtFinType1) & (is.na(all$BsmtCond) | is.na(all$BsmtQual) | 
                                  is.na(all$BsmtExposure) | is.na(all$BsmtFinType2)), 
    c('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2')]

# 최빈도값으로 대체값 할당
all$BsmtFinType2[333] <- names(sort(-table(all$BsmtFinType2)))[1]
all$BsmtExposure[c(949, 1488, 2349)] <- names(sort(-table(all$BsmtExposure)))[1]
all$BsmtQual[c(2218,2219)] <- names(sort(-table(all$BsmtQual)))[1]
all$BsmtCond[c(2041, 2186, 2525)] <- names(sort(-table(all$BsmtCond)))[1]

all$BsmtQual[is.na(all$BsmtQual)] <- 'None'
all$BsmtQual <- as.integer(revalue(all$BsmtQual, Qualities))
table(all$BsmtQual)

all$BsmtCond[is.na(all$BsmtCond)] <- 'None'
all$BsmtCond <- as.integer(revalue(all$BsmtCond, Qualities))
table(all$BsmtCond)

all$BsmtExposure[is.na(all$BsmtExposure)] <- 'None'
Exposure <- c('None' = 0, 'No' = 1, 'Mn' = 2, 'Av' = 3, 'Gd' = 4)

all$BsmtExposure <- as.integer(revalue(all$BsmtExposure, Exposure))
table(all$BsmtExposure)

all$BsmtFinType1[is.na(all$BsmtFinType1)] <- 'None'
Fintype <- c('None' = 0, 'Unf' = 1, 'LwQ' = 2, 'Rec' = 3, 'BLQ' = 4, 'ALQ' = 5, 'GLQ' = 6)

all$BsmtFinType1 <- as.integer(revalue(all$BsmtFinType1, Fintype))
table(all$BsmtFinType1)

all$BsmtFinType2[is.na(all$BsmtFinType2)] <- 'None'
all$BsmtFinType2 <- as.integer(revalue(all$BsmtFinType2, Fintype))
table(all$BsmtFinType2)

# 상기에 관측했던 지하실이 없었던 79채를 참고하여 남은 결측치를 확인해보자
all[(is.na(all$BsmtFullBath) | is.na(all$BsmtHalfBath) | is.na(all$BsmtFinSF1) | 
       is.na(all$BsmtFinSF2) | is.na(all$BsmtUnfSF) | is.na(all$TotalBsmtSF)), 
    c('BsmtQual', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF')]

all$BsmtFullBath[is.na(all$BsmtFullBath)] <- 0
table(all$BsmtFullBath)

all$BsmtHalfBath[is.na(all$BsmtHalfBath)] <- 0
table(all$BsmtHalfBath)

all$BsmtFinSF1[is.na(all$BsmtFinSF1)] <- 0
all$BsmtFinSF2[is.na(all$BsmtFinSF2)] <- 0
all$BsmtUnfSF[is.na(all$BsmtUnfSF)] <- 0
all$TotalBsmtSF[is.na(all$TotalBsmtSF)] <- 0

#Masonry veneer type, and masonry veneer area 
# veneer area의 23개 결측치가 veneer type의 23개 결측치인지 확인해 보겠다. 
length(which(is.na(all$MasVnrType) & is.na(all$MasVnrArea)))

# veneer type의 1개 결측치를 찾아보자.
all[is.na(all$MasVnrType) & !is.na(all$MasVnrArea), c('MasVnrType', 'MasVnrArea')]

#veneer type의 결측치를 최빈도값으로 대체하자
all$MasVnrType[2611] <- names(sort(-table(all$MasVnrType)))[2] #최빈도는 'none'이라 두번째 빈도로 했다.
all[2611, c('MasVnrType', 'MasVnrArea')]

all$MasVnrType[is.na(all$MasVnrType)] <- 'None'
all[!is.na(all$SalePrice),] %>%    #결측치 아닌 SalePrice 한정
  group_by(MasVnrType) %>%         # MasVnrType 변수로 그룹핑
  summarise(median = median(SalePrice), counts=n()) %>%  # SalePrice 중위값, 개수 요약
  arrange(median)                  # 중위값 순으로 오름차순 정렬

Masonry <- c('None' = 0, 'BrkCmn' = 0, 'BrkFace' = 1, 'Stone' = 2)
all$MasVnrType <- as.integer(revalue(all$MasVnrType, Masonry))
table(all$MasVnrType)

all$MasVnrArea[is.na(all$MasVnrArea)] <- 0

#MSZoning: 용도별 지구 식별자 
# 최빈도값으로 결측치 대체
all$MSZoning[is.na(all$MSZoning)] <- names(sort(-table(all$MSZoning)))[1]
all$MSZoning <- as.factor(all$MSZoning)
table(all$MSZoning)

sum(table(all$MSZoning))

# Kitchen quality and number of Kitchens above grade
# 최빈도값으로 결측치 대체
all$KitchenQual[is.na(all$KitchenQual)] <- 'TA' 
all$KitchenQual <- as.integer(revalue(all$KitchenQual, Qualities))
table(all$KitchenQual)
sum(table(all$KitchenQual))
table(all$KitchenAbvGr)
sum(table(all$KitchenAbvGr))

#Utilities: 사용할 수 있는 Utilities의 종류
table(all$Utilities)
kable(all[is.na(all$Utilities) | all$Utilities == "NoSeWa", 1:9])
all$Utilities <- NULL


# Functional: 홈 기능
# 최빈도값으로 결측치 대체
all$Functional[is.na(all$Functional)] <- names(sort(-table(all$Functional)))[1]
all$Functional <- as.integer(revalue(all$Functional, +
                                       c('Sal' = 0, 'Sev' = 1, 'Maj2' = 2, 'Maj1' = 3, 'Mod' = 4, 'Min2' = 5, 'Min1' = 6, 'Typ' = 7)))
table(all$Functional)
sum(table(all$Functional))

#건물 외장 변수
# 최빈도값으로 결측치 대체
all$Exterior1st[is.na(all$Exterior1st)] <- names(sort(-table(all$Exterior1st)))[1]
all$Exterior1st <- as.factor(all$Exterior1st)
table(all$Exterior1st)
sum(table(all$Exterior1st))

# 최빈도값으로 결측치 대체
all$Exterior2nd[is.na(all$Exterior2nd)] <- names(sort(-table(all$Exterior2nd)))[1]
all$Exterior2nd <- as.factor(all$Exterior2nd)
table(all$Exterior2nd)
sum(table(all$Exterior2nd))

all$ExterQual <- as.integer(revalue(all$ExterQual, Qualities))
table(all$ExterQual)
sum(table(all$ExterQual))

all$ExterCond <- as.integer(revalue(all$ExterCond, Qualities))
table(all$ExterCond)
sum(table(all$ExterCond))

# Electrical: 전기 시스템
# 최빈도값으로 결측치 대체
all$Electrical[is.na(all$Electrical)] <- names(sort(-table(all$Electrical)))[1]
all$Electrical <- as.factor(all$Electrical)
table(all$Electrical)

sum(table(all$Electrical))

#SaleType: 판매 방식
# 최빈도값으로 결측치 대체
all$SaleType[is.na(all$SaleType)] <- names(sort(-table(all$SaleType)))[1]
all$SaleType <- as.factor(all$SaleType)
table(all$SaleType)
sum(table(all$SaleType))
#SaleCondition: 판매 조건
all$SaleCondition <- as.factor(all$SaleCondition)
table(all$SaleCondition)
sum(table(all$SaleCondition))



# 문자형 변수
Charcol <- names(all[,sapply(all, is.character)]) #문자형 변수만 선별하여 생성
cat('There are', length(Charcol), 'remaining columns with character values')

#Foundation: 건물 기초(토대)의 종류
#순서형이 아니기에, factor형으로 변환하겠다.
all$Foundation <- as.factor(all$Foundation)
table(all$Foundation)
sum(table(all$Foundation))

#순서형이 아니기에, factor형으로 변환하겠다.
all$Heating <- as.factor(all$Heating)
table(all$Heating)
sum(table(all$Heating))

# Qualities 벡터로 순서형으로 변환한다.
all$HeatingQC <- as.integer(revalue(all$HeatingQC, Qualities))
table(all$HeatingQC)
sum(table(all$HeatingQC))

all$CentralAir <- as.integer(revalue(all$CentralAir, c('N' = 0, 'Y' = 1)))
table(all$CentralAir)
sum(table(all$CentralAir))

# RoofStyle: 지붕의 종류
# 순서형이 아니기에, factor형으로 변환하겠다.
all$RoofStyle <- as.factor(all$RoofStyle)
table(all$RoofStyle)
sum(table(all$RoofStyle))
#RoofMatl: 지붕 재료
# 순서형이 아니기에, factor형으로 변환하겠다.
all$RoofMatl <- as.factor(all$RoofMatl)
table(all$RoofMatl)
sum(table(all$RoofMatl))

#  LandContour: 부지의 평탄함
# 순서형이 아니기에, factor형으로 변환하겠다.
all$LandContour <- as.factor(all$LandContour)
table(all$LandContour)
sum(table(all$LandContour))
#LandSlope: 부지의 경사(비탈)
# 순서형 타입, 정수형으로 변환하겠다.
all$LandSlope <- as.integer(revalue(all$LandSlope, c('Sev' = 0, 'Mod' = 1, 'Gtl' = 2)))
table(all$LandSlope)
sum(table(all$LandSlope))

#BldgType: 주거의 형태
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(BldgType), y = SalePrice)) +
  geom_bar(stat = 'summary', fun.y = 'median', fill = 'blue') +
  scale_y_continuous(breaks = seq(0, 800000, by = 100000), labels = comma) +
  geom_label(stat = 'count', aes(label = ..count.., y = ..count..))

# 순서형 범주가 아니기에, factor로 변환한다.
all$BldgType <- as.factor(all$BldgType)
table(all$BldgType)
sum(table(all$BldgType))

# HouseStyle: 주거의 Style
# 순서형 범주가 아니기에, factor로 변환한다.
all$HouseStyle <- as.factor(all$HouseStyle)
table(all$HouseStyle)
sum(table(all$HouseStyle))

#물리적 거리, 근방의 입지에 따른 3개의 변수
# 순서형 범주가 아니기에, factor로 변환한다.
all$Neighborhood <- as.factor(all$Neighborhood)
table(all$Neighborhood)
sum(table(all$Neighborhood))
# Condition1: 근방의 다양한 조건
# 순서형 범주가 아니기에, factor로 변환한다.
all$Condition1 <- as.factor(all$Condition1)
table(all$Condition1)
sum(table(all$Condition1))
#Condition2: 근방의 다양한 조건 (2개 이상)
# 순서형 범주가 아니기에, factor로 변환한다.
all$Condition2 <- as.factor(all$Condition2)
table(all$Condition2)
sum(table(all$Condition2))
#Street: 부지에 접한 길의 종류
# 순서형 범주로, 정수형으로 변환하겠다.
all$Street <- as.integer(revalue(all$Street, c('Grvl' = 0, 'Pave' = 1)))
table(all$Street)
sum(table(all$Street))
#PavedDrive: 진입로의 포장
#순서형 범주로, 정수형으로 변환하겠다.
all$PavedDrive <- as.integer(revalue(all$PavedDrive, c('N' = 0, 'P' = 1, 'Y' = 2)))
table(all$PavedDrive)
sum(table(all$PavedDrive))

#numeric variable
str(all$YrSold)
str(all$MoSold)
all$MoSold <- as.factor(all$MoSold)


ys <- ggplot(all[!is.na(all$SalePrice),], aes(x = as.factor(YrSold), y = SalePrice)) +
  geom_bar(stat = 'summary', fun.y = 'median', fill = 'blue') +
  scale_y_continuous(breaks = seq(0, 800000, by = 25000), labels = comma) +
  geom_label(stat = 'count', aes(label = ..count.., y = ..count..)) +
  coord_cartesian(ylim = c(0, 200000)) + #y축 20만까지 표기 제한
  geom_hline(yintercept = 163000, linetype='dashed', color = 'red') #SalePrice 중위값

ms <- ggplot(all[!is.na(all$SalePrice),], aes(x = MoSold, y = SalePrice)) + 
  geom_bar(stat = 'summary', fun.y = 'median', fill = 'blue') +
  scale_y_continuous(breaks = seq(0, 800000, by = 25000), labels = comma) + 
  geom_label(stat = 'count', aes(label = ..count.., y = ..count..)) + 
  coord_cartesian(ylim = c(0, 200000)) +
  geom_hline(yintercept = 163000, linetype = 'dashed', color = 'red') 

library(gridExtra)
grid.arrange(ys, ms, widths = c(1,2))


#MSSubClass: 판매와 연관된 주거 타입

str(all$MSSubClass)

all$MSSubClass <- as.factor(all$MSSubClass)

# 가독성을 높이기 위해 숫자를 문자로 revalue
all$MSSubClass <- revalue(all$MSSubClass, c('20'='1 story 1946+', '30'='1 story 1945-', '40'='1 story unf attic', '45'='1,5 story unf', '50'='1,5 story fin', '60'='2 story 1946+', '70'='2 story 1945-', '75'='2,5 story all ages', '80'='split/multi level', '85'='split foyer', '90'='duplex all style/age', '120'='1 story PUD 1946+', '150'='1,5 story PUD all', '160'='2 story PUD 1946+', '180'='PUD multilevel', '190'='2 family conversion'))

str(all$MSSubClass)

numericVars <- which(sapply(all, is.numeric)) # index vector numeric variables
factorVars <- which(sapply(all, is.factor))   # index vector factor variables
cat('There are', length(numericVars), 'numeric variables, and', length(factorVars), 
    'categoric variables')

all_numVar <- all[, numericVars]
cor_numVar <- cor(all_numVar, use = 'pairwise.complete.obs') # 모든 수치형 변수의 상관 계수

# SalePrice와 변수들의 상관 계수 내림차순 정렬
cor_sorted <- as.matrix(sort(cor_numVar[, 'SalePrice'], decreasing = TRUE))
# 0.5보다 높은 상관 계수 취사 선택
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x) > 0.5)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]

corrplot.mixed(cor_numVar, 
               tl.col = 'black',  # 변수명 색깔
               tl.pos = 'lt',     # 변수명 왼쪽 표시
               tl.cex = 0.7,      # 변수명 text 크기
               cl.cex = 0.7,      # y축 상관계수 text 크기
               number.cex = .7    # matrix안 상관계수 text 크기
)

library(randomForest)
set.seed(2018) #인수 '2018'로 시드 생성
quick_RF <- randomForest(x = all[1:1460, -79], y = all$SalePrice[1:1460], ntree = 100, importance = TRUE)
imp_RF <- importance(quick_RF)
imp_DF <- data.frame(Variables = row.names(imp_RF), MSE = imp_RF[,1])
imp_DF <- imp_DF[order(imp_DF$MSE, decreasing = TRUE),]

ggplot(imp_DF[1:20,],
       aes(x = reorder(Variables, MSE), y = MSE, fill = MSE)) + # MSE기준 변수 재정렬
  geom_bar(stat = 'identity') + 
  labs(x = 'Variables', y = '% increase MSE if variable is randomly permuted') + #x,y축명 명명
  coord_flip() + #x, y축 반전
  theme(legend.position = 'none')


s1 <- ggplot(data = all, aes(x = GrLivArea)) +
  geom_density() + labs(x = 'Square feet living area')
s2 <- ggplot(data = all, aes(x = as.factor(TotRmsAbvGrd))) +
  geom_histogram(stat = 'count') + labs(x = 'Rooms above Ground')
s3 <- ggplot(all, aes(x = X1stFlrSF)) +
  geom_density() + labs(x = 'Square feet first floor')
s4 <- ggplot(all, aes(x = X2ndFlrSF)) + 
  geom_density() + labs(x = 'Square feet second floor')
s5 <- ggplot(all, aes(x = TotalBsmtSF)) +
  geom_density()+ labs(x = 'Square feet basement')
s6 <- ggplot(all[all$LotArea < 100000,], aes(x = LotArea)) + 
  geom_density() + labs(x = 'Square feet lot')
s7 <- ggplot(all, aes(x = LotFrontage)) +
  geom_density() + labs(x = 'Linear feet lot frontage')
s8 <- ggplot(all, aes(x = LowQualFinSF)) +
  geom_histogram() + labs(x = 'Low quality square feet 1st & 2nd')

layout <- matrix(c(1,2,5,3,4,8,6,7),4,2, byrow = TRUE) # 4행 2열 ()안의 순으로 행부터 matrix 생성
multiplot(s1, s2, s3, s4, s5, s6, s7, s8, layout = layout)

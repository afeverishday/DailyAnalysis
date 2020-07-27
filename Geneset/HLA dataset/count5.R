library(readxl)
library(dplyr)

rm(list = ls())
gc()

# 나라별 면역 데이터를 대륙별로 통계


Africa = c('Morocco','Kenya','South Africa','Tunisia','Nigeria')

Asia = c('China','Sri Lanka', 'Hong Kong','India','Iran','Israel',
         'Saudi Arabia','Malaysia','Pakistan','South Korea','Taiwan','Thailand','Turkey')

Europe = c('Russia','Finland','Czech Republic','France','Germany','Greece',
           'Italy','Netherlands','Poland','Spain','Sweden')

#'Serbia 데이터는 location이 생략되어 있음.

NorthAmerica= c('Jamaica','USA')
Oceania=c('Australia')
SouthAmerica=c('Brazil','Colombia','Peru')


for (i in 1:length(Africa)){
  print(i)
  assign(Africa[i] ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", 
                               sheet = Africa[i], 
                               skip = 2,
                               col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample',	'Location'),
                               col_types = "guess", # guess the types of columns
                               na = "NA" )%>%  dplyr::filter((Sample>=50)&(Frequency!=0)&!is.na(Frequency))%>%
           dplyr::select(-Line,-individuals,-Population, -Location)%>% 
           dplyr::filter(substr(Allele,1,1)!='D') %>%
           mutate(Country=Africa[i],
                  ncount= floor(Sample*2*Frequency)) %>%  dplyr::arrange(Allele, desc(Frequency)) %>%
           filter(ncount!=0) %>% dplyr::group_by(Allele) %>% dplyr::summarise(Ncount= sum(ncount),                                                                                                                Country=unique(Country))) 
}






for (i in 1:length(Asia)) {
  print(i)
  assign(Asia[i] ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                             sheet = Asia[i], # sheet name to read from
                             skip = 2,
                             col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample',	'Location'), 
                             col_types = "guess", # guess the types of columns
                             na = "NA" )%>%  dplyr::filter((Sample>=50)&(Frequency!=0)&!is.na(Frequency))%>%
           dplyr::select(-Line,-individuals,-Population, -Location)%>% 
           dplyr::filter(substr(Allele,1,1)!='D') %>%
           mutate(Country=Asia[i],
                  ncount= floor(Sample*2*Frequency))%>%  dplyr::arrange(Allele, desc(Frequency)) %>%
           filter(ncount!=0) %>% dplyr::group_by(Allele) %>% dplyr::summarise(Ncount= sum(ncount),                                                                                                                Country=unique(Country))) 
}



for (i in 1:length(Europe)) {
  print(i)
  assign(Europe[i] ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                               sheet = Europe[i], # sheet name to read from
                               skip = 2,
                               col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample',	'Location'),
                               col_types = "guess", # guess the types of columns
                               na = "NA" )%>%  dplyr::filter((Sample>=50)&(Frequency!=0)&!is.na(Frequency))%>%
           dplyr::select(-Line,-individuals,-Population, -Location)%>% 
           dplyr::filter(substr(Allele,1,1)!='D') %>%
           mutate(Country=Europe[i],
                  ncount= floor(Sample*2*Frequency))%>%  dplyr::arrange(Allele, desc(Frequency)) %>%
           filter(ncount!=0) %>% dplyr::group_by(Allele) %>% dplyr::summarise(Ncount= sum(ncount),                                                                                                                Country=unique(Country))) 
}

# Serbia는 location이 존재하지 않음.
assign("Serbia" ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                            sheet = "Serbia", # sheet name to read from
                            skip = 2,
                            col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample'), 
                            col_types = "guess", # guess the types of columns
                            na = "NA" )%>%  dplyr::filter((Sample>=50)&(Frequency!=0)&!is.na(Frequency))%>%
         dplyr::select(-Line,-individuals,-Population)%>% 
         dplyr::filter(substr(Allele,1,1)!='D') %>%
         mutate(Country=Europe[i],
                ncount= floor(Sample*2*Frequency))%>%  dplyr::arrange(Allele, desc(Frequency)) %>%
         filter(ncount!=0) %>% dplyr::group_by(Allele) %>% dplyr::summarise(Ncount= sum(ncount),                                                                                                                Country=unique(Country))) 



for (i in 1:length(NorthAmerica)) {
  print(i)
  assign(NorthAmerica[i] ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                                     sheet = NorthAmerica[i], # sheet name to read from
                                     skip = 2,
                                     col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample',	'Location'),
                                     col_types = "guess", # guess the types of columns
                                     na = "NA" )%>%  dplyr::filter((Sample>=50)&(Frequency!=0)&!is.na(Frequency))%>%
           dplyr::select(-Line,-individuals,-Population, -Location)%>% 
           dplyr::filter(substr(Allele,1,1)!='D') %>%
           mutate(Country=NorthAmerica[i],
                  ncount= floor(Sample*2*Frequency))%>% dplyr::arrange(Allele, desc(Frequency)) %>%
           filter(ncount!=0) %>% dplyr::group_by(Allele) %>% dplyr::summarise(Ncount= sum(ncount),                                                                                                                Country=unique(Country))) 
}



for (i in 1:length(Oceania)) {
  print(i)
  assign(Oceania[i] ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                                sheet = Oceania[i], # sheet name to read from
                                skip = 2,
                                col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample',	'Location'),
                                col_types = "guess", # guess the types of columns
                                na = "NA" )%>%  dplyr::filter((Sample>=50)&(Frequency!=0)&!is.na(Frequency))%>%
           dplyr::select(-Line,-individuals,-Population, -Location)%>% 
           dplyr::filter(substr(Allele,1,1)!='D') %>%
           mutate(Country=Oceania[i],
                  ncount= floor(Sample*2*Frequency))%>% dplyr::arrange(Allele, desc(Frequency)) %>%
           filter(ncount!=0) %>% dplyr::group_by(Allele) %>% dplyr::summarise(Ncount= sum(ncount),                                                                                                                Country=unique(Country))) 
}




for (i in 1:length(SouthAmerica)) {
  print(i)
  assign(SouthAmerica[i] ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                                     sheet = SouthAmerica[i], # sheet name to read from
                                     skip = 2,
                                     col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample',	'Location'), 
                                     col_types = "guess", # guess the types of columns
                                     na = "NA" )%>%  dplyr::filter((Sample>=50)&(Frequency!=0)&!is.na(Frequency))%>%
           dplyr::select(-Line,-individuals,-Population, -Location)%>% 
           dplyr::filter(substr(Allele,1,1)!='D') %>%
           mutate(Country=SouthAmerica[i],
                  ncount= floor(Sample*2*Frequency))%>% dplyr::arrange(Allele, desc(Frequency)) %>%
           filter(ncount!=0) %>% dplyr::group_by(Allele) %>% dplyr::summarise(Ncount= sum(ncount),                                                                                                                Country=unique(Country))) 
}




tot1<-bind_rows(Morocco, Kenya, Nigeria, `South Africa`, Tunisia,
                China,`Sri Lanka`,`Hong Kong`,India,Iran, Israel, `Saudi Arabia`, Malaysia, Pakistan,`South Korea`, Taiwan, Thailand, Turkey,
                Russia, Finland, `Czech Republic`, France, Germany,Greece, Italy, Netherlands, Poland, Serbia, Spain,  Sweden,
                Jamaica, USA,
                Australia,
                Brazil, Colombia, Peru)  %>% 
  dplyr::group_by(Allele) %>% dplyr::summarise(tot_count= sum(Ncount))

tot1$tot_Frequency<- tot1$tot_count/sum(tot1$tot_count)


tot2<-full_join(Morocco , Kenya, by=c('Allele')) %>%
  full_join(`South Africa`, by=c('Allele')) %>% full_join(Tunisia, by=c('Allele')) %>%
  full_join( China, by=c('Allele')) %>% full_join( `Sri Lanka`, by=c('Allele')) %>%
  full_join(`Hong Kong` , by=c('Allele')) %>% full_join(India , by=c('Allele')) %>%
  full_join(Iran , by=c('Allele')) %>% full_join( Israel, by=c('Allele')) %>%
  full_join(`Saudi Arabia` , by=c('Allele')) %>% full_join(Malaysia , by=c('Allele')) %>%
  full_join(Pakistan , by=c('Allele')) %>% full_join(`South Korea` , by=c('Allele')) %>%
  full_join(Taiwan , by=c('Allele')) %>% full_join( Thailand, by=c('Allele'))%>%
  mutate(Country= paste0(ifelse(is.na(Ncount.x),'',paste0(Country.x, ': ', Ncount.x, ' ') ),
                         ifelse(is.na(Ncount.y),'',paste0(Country.y, ': ', Ncount.y, ' ') ), 
                         ifelse(is.na(Ncount.x.x),'',paste0(Country.x.x,': ', Ncount.x.x, ' ') ), 
                         ifelse(is.na(Ncount.y.y),'',paste0(Country.y.y,': ', Ncount.y.y, ' ') ), 
                         ifelse(is.na(Ncount.x.x.x),'',paste0(Country.x.x.x,': ',Ncount.x.x.x, ' ' ) ), 
                         ifelse(is.na(Ncount.y.y.y),'',paste0(Country.y.y.y,': ', Ncount.y.y.y, ' ') ), 
                         ifelse(is.na(Ncount.x.x.x.x),'',paste0(Country.x.x.x.x,': ', Ncount.x.x.x.x, ' ') ), 
                         ifelse(is.na(Ncount.y.y.y.y),'',paste0(Country.y.y.y.y,': ', Ncount.y.y.y.y, ' ') ), 
                         ifelse(is.na(Ncount.x.x.x.x.x),'',paste0(Country.x.x.x.x.x,': ', Ncount.x.x.x.x.x, ' ') ), 
                         ifelse(is.na(Ncount.y.y.y.y.y),'',paste0(Country.y.y.y.y.y,': ', Ncount.y.y.y.y.y, ' ') ), 
                         ifelse(is.na(Ncount.x.x.x.x.x.x),'',paste0(Country.x.x.x.x.x.x,': ',Ncount.x.x.x.x.x.x, ' ' ) ),
                         ifelse(is.na(Ncount.y.y.y.y.y.y),'',paste0(Country.y.y.y.y.y.y,': ',Ncount.y.y.y.y.y.y, ' ' ) ), 
                         ifelse(is.na(Ncount.x.x.x.x.x.x.x),'',paste0(Country.x.x.x.x.x.x.x,': ',Ncount.x.x.x.x.x.x.x, ' ' ) ),
                         ifelse(is.na(Ncount.y.y.y.y.y.y.y),'',paste0(Country.y.y.y.y.y.y.y,': ', Ncount.y.y.y.y.y.y.y, ' ') ),
                         ifelse(is.na(Ncount.x.x.x.x.x.x.x.x),'',paste0(Country.x.x.x.x.x.x.x.x,': ',Ncount.x.x.x.x.x.x.x.x, ' ' ) ),
                         ifelse(is.na(Ncount.y.y.y.y.y.y.y.y),'',paste0(Country.y.y.y.y.y.y.y.y,': ',Ncount.y.y.y.y.y.y.y.y, ' ' ) )
  ))  %>% dplyr::select(Allele,Country )


tot3 <- full_join( Turkey, Russia , by=c('Allele')) %>% 
  full_join(Finland , by=c('Allele')) %>% full_join(`Czech Republic` , by=c('Allele')) %>%
  full_join(France , by=c('Allele')) %>% full_join(Germany , by=c('Allele')) %>%
  full_join(Greece , by=c('Allele')) %>% full_join(Italy , by=c('Allele')) %>%
  full_join(Netherlands , by=c('Allele')) %>% full_join(Poland , by=c('Allele')) %>%
  full_join(Serbia , by=c('Allele')) %>% full_join(Spain , by=c('Allele')) %>%
  full_join(Sweden , by=c('Allele')) %>% full_join(USA , by=c('Allele')) %>%
  full_join(Australia, by=c('Allele')) %>% full_join(Brazil , by=c('Allele')) %>% 
  full_join( Colombia, by=c('Allele')) %>% full_join(Peru, by=c('Allele'))%>%
  mutate(Country= paste0(ifelse(is.na(Ncount.x),'',paste0(Country.x, ': ', Ncount.x, ' ') ),
                         ifelse(is.na(Ncount.y),'',paste0(Country.y, ': ', Ncount.y, ' ') ), 
                         ifelse(is.na(Ncount.x.x),'',paste0(Country.x.x,': ', Ncount.x.x, ' ') ), 
                         ifelse(is.na(Ncount.y.y),'',paste0(Country.y.y,': ', Ncount.y.y, ' ') ), 
                         ifelse(is.na(Ncount.x.x.x),'',paste0(Country.x.x.x,': ',Ncount.x.x.x, ' ' ) ), 
                         ifelse(is.na(Ncount.y.y.y),'',paste0(Country.y.y.y,': ', Ncount.y.y.y, ' ') ), 
                         ifelse(is.na(Ncount.x.x.x.x),'',paste0(Country.x.x.x.x,': ', Ncount.x.x.x.x, ' ') ), 
                         ifelse(is.na(Ncount.y.y.y.y),'',paste0(Country.y.y.y.y,': ', Ncount.y.y.y.y, ' ') ), 
                         ifelse(is.na(Ncount.x.x.x.x.x),'',paste0(Country.x.x.x.x.x,': ', Ncount.x.x.x.x.x, ' ') ), 
                         ifelse(is.na(Ncount.y.y.y.y.y),'',paste0(Country.y.y.y.y.y,': ', Ncount.y.y.y.y.y, ' ') ), 
                         ifelse(is.na(Ncount.x.x.x.x.x.x),'',paste0(Country.x.x.x.x.x.x,': ',Ncount.x.x.x.x.x.x, ' ' ) ),
                         ifelse(is.na(Ncount.y.y.y.y.y.y),'',paste0(Country.y.y.y.y.y.y,': ',Ncount.y.y.y.y.y.y, ' ' ) ), 
                         ifelse(is.na(Ncount.x.x.x.x.x.x.x),'',paste0(Country.x.x.x.x.x.x.x,': ',Ncount.x.x.x.x.x.x.x, ' ' ) ),
                         ifelse(is.na(Ncount.y.y.y.y.y.y.y),'',paste0(Country.y.y.y.y.y.y.y,': ', Ncount.y.y.y.y.y.y.y, ' ') ),
                         ifelse(is.na(Ncount.x.x.x.x.x.x.x.x),'',paste0(Country.x.x.x.x.x.x.x.x,': ',Ncount.x.x.x.x.x.x.x.x, ' ' ) ),
                         ifelse(is.na(Ncount.y.y.y.y.y.y.y.y),'',paste0(Country.y.y.y.y.y.y.y.y,': ',Ncount.y.y.y.y.y.y.y.y, ' ' ) ),
                         ifelse(is.na(Ncount.x.x.x.x.x.x.x.x.x),'',paste0(Country.x.x.x.x.x.x.x.x.x,': ',Ncount.x.x.x.x.x.x.x.x.x, ' ' ) ),
                         ifelse(is.na(Ncount.y.y.y.y.y.y.y.y.y),'',paste0(Country.y.y.y.y.y.y.y.y.y,': ',Ncount.y.y.y.y.y.y.y.y.y, ' ' ) )
  ))  %>% dplyr::select(Allele,Country )


tot_cont <- full_join( tot2, tot3 , by=c('Allele')) %>% mutate(Country=paste0(ifelse(is.na(Country.x),'', Country.x), ifelse(is.na(Country.y),'',Country.y ))) %>% select(Allele, Country)%>% arrange(Allele)


result<- full_join(tot1, tot_cont, by=c('Allele'))%>% arrange(Allele) 


a=result %>% filter(substr(Allele, 1,1)=='A') %>% summarise(sum(tot_count)) %>% as.vector()
b=result %>% filter(substr(Allele, 1,1)=='B') %>% summarise(sum(tot_count))
c=result %>% filter(substr(Allele, 1,1)=='C') %>% summarise(sum(tot_count))

result %>% summarise(sum(tot_count))
  

result_a<- result %>% filter(substr(Allele, 1,1)=='A') %>% mutate(Allele_Frequency= (tot_count/7305241)) %>% select(-tot_Frequency)
result_b<- result %>% filter(substr(Allele, 1,1)=='B') %>% mutate(Allele_Frequency= (tot_count/7315343)) %>% select(-tot_Frequency)
result_c<- result %>% filter(substr(Allele, 1,1)=='C') %>% mutate(Allele_Frequency= (tot_count/5492177)) %>% select(-tot_Frequency)

result_tot=bind_rows(result_a, result_b, result_c)%>% select(Allele, tot_count, Allele_Frequency, Country) %>% arrange(Allele)
names(result_tot)<- c("Allele", "Allele_Count", "Allele_Frequency", "Country" )

write.table(result_tot, "C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/output9.txt",
            sep = "\t",
            row.names = F)




# Frequency가 0이 아닌데 count가 0이 되는 경우
# Kenya%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
# `South Africa`%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
# Tunisia%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
# China%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
# `Hong Kong`%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
# India%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
# Iran%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
# Israel%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
# Malaysia%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
# Pakistan%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
# `Saudi Arabia`%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
# `South Korea`%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
# Taiwan%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
# Thailand%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
# Turkey%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()

# Germany%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
# Greece%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
# Italy%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
# Netherlands%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
# Russia%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
# Spain%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
# Sweden%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()

# USA%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
# Brazil%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table() 
# Colombia%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()



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


for (i in 1:length(Africa)) {
  print(i)
  assign(Africa[i] ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                               sheet = Africa[i], # sheet name to read from
                               skip = 2,
                               col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample',	'Location'), # TRUE to use the first row as column names
                               col_types = "guess", # guess the types of columns
                               na = "NA" )%>%  dplyr::filter((Frequency>=0.01)&(Sample>=50))%>%
           dplyr::select(-Line,-individuals,-Population, -Location)%>% 
           mutate(Country=Africa[i],
                  Type=ifelse(substr(Allele,1,1)=='A', 'A',
                              ifelse(substr(Allele,1,1)=='B', 'B',
                                     ifelse(substr(Allele,1,1)=='C', 'C',
                                            ifelse(substr(Allele,1,2)=='DQ', 'DQ',
                                                   ifelse(substr(Allele,1,2)=='DP', 'DP',
                                                          ifelse(substr(Allele,1,2)=='DR', 'DR',substr(Allele,1,2))
                                                   )
                                            )
                                     )
                              )
                  ),
                  Geo='Africa')%>% dplyr::filter(Type %in% c('A', 'B', 'C'))  %>% dplyr::select(-Sample, -Type) %>% dplyr::arrange(Allele, Frequency)) 
}

Morocco = Morocco[!duplicated(Morocco[,c('Allele')]),]
Kenya = Kenya[!duplicated(Kenya[,c('Allele')]),] 
Nigeria = Nigeria[!duplicated(Nigeria[,c('Allele')]),]
`South Africa`= `South Africa`[!duplicated(`South Africa`[,c('Allele')]),] 
Tunisia = Tunisia[!duplicated(Tunisia[,c('Allele')]),] 

colSums(is.na(Morocco))
colSums(is.na(Kenya))
colSums(is.na(Nigeria))
colSums(is.na(`South Africa`))
colSums(is.na(Tunisia))


Africa_new1<- bind_rows(Morocco, Kenya, Nigeria, `South Africa`, Tunisia) %>% 
  dplyr::select( Allele, Country, Geo,Frequency) %>% mutate(count=1) %>% arrange(Allele, Country)


Africa_new2<- full_join(Morocco %>% dplyr::select(Allele, Country), Kenya%>% dplyr::select(Allele, Country), by=c('Allele')) %>%
  full_join(`South Africa`%>% dplyr::select(Allele, Country), by=c('Allele')) %>%
  full_join(Tunisia%>% dplyr::select(Allele, Country), by=c('Allele')) %>% arrange(Allele)
  
Africa_new2<- Africa_new2 %>% mutate(INFO= paste0( 'Count: ',  ifelse( !is.na(Country.x), 1, 0)+ifelse( !is.na(Country.y), 1, 0)+
                       ifelse( !is.na(Country.x.x), 1, 0)+ifelse( !is.na(Country.y.y), 1, 0)
                       ,'/',length(Africa)-1, ', Country: ', 
                       ifelse( !is.na(Country.x), 'Morocco ', ''),ifelse( !is.na(Country.y), 'Kenya ', ''),
                       ifelse( !is.na(Country.x.x), 'South Africa ', ''), ifelse( !is.na(Country.y.y), 'Tunisia ', '')))%>% 
  dplyr::select(Allele, INFO)


Africa_tot<-dplyr::full_join(Africa_new1 , Africa_new2, by=c('Allele') ) %>% 
  dplyr::select(Country, Geo, Allele, Frequency, INFO)

colnames(Africa_tot)<- c('Country', 'Geo', 'HLA_type', 'Frequency', 'INFO')








for (i in 1:length(Asia)) {
  print(i)
  assign(Asia[i] ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                             sheet = Asia[i], # sheet name to read from
                             skip = 2,
                             col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample',	'Location'), # TRUE to use the first row as column names
                             col_types = "guess", # guess the types of columns
                             na = "NA" )%>%  dplyr::filter((Frequency>=0.01)&(Sample>=50))%>%
           dplyr::select(-Line,-individuals,-Population, -Location)%>% 
           mutate(Country=Asia[i],
                  Type=ifelse(substr(Allele,1,1)=='A', 'A',
                              ifelse(substr(Allele,1,1)=='B', 'B',
                                     ifelse(substr(Allele,1,1)=='C', 'C',
                                            ifelse(substr(Allele,1,2)=='DQ', 'DQ',
                                                   ifelse(substr(Allele,1,2)=='DP', 'DP',
                                                          ifelse(substr(Allele,1,2)=='DR', 'DR',substr(Allele,1,2))
                                                   )
                                            )
                                     )
                              )
                  ),
                  Geo='Asia')%>% dplyr::filter(Type %in% c('A', 'B', 'C'))%>% dplyr::select(-Sample, -Type) %>% dplyr::arrange(Allele, Frequency)) 
}

China = China[!duplicated(China[,c('Allele')]),] 
`Sri Lanka` = `Sri Lanka`[!duplicated(`Sri Lanka`[,c('Allele')]),] 
`Hong Kong` = `Hong Kong`[!duplicated(`Hong Kong`[,c('Allele')]),] 
India = India[!duplicated(India[,c('Allele')]),] 
Iran = Iran[!duplicated(Iran[,c('Allele')]),] 
Israel = Israel[!duplicated(Israel[,c('Allele')]),] 
`Saudi Arabia` = `Saudi Arabia`[!duplicated(`Saudi Arabia`[,c('Allele')]),] 
Malaysia = Malaysia[!duplicated(Malaysia[,c('Allele')]),] 
Pakistan = Pakistan[!duplicated(Pakistan[,c('Allele')]),] 
`South Korea` = `South Korea`[!duplicated(`South Korea`[,c('Allele')]),] 
Taiwan = Taiwan[!duplicated(Taiwan[,c('Allele')]),] 
Thailand = Thailand[!duplicated(Thailand[,c('Allele')]),] 
Turkey = Turkey[!duplicated(Turkey[,c('Allele')]),] 


Asia_new1<- bind_rows(China,`Sri Lanka`,`Hong Kong`,India,Iran, Israel, `Saudi Arabia`, Malaysia,
                     Pakistan,`South Korea`, Taiwan, Thailand, Turkey ) %>% 
  dplyr::select( Allele, Country, Geo, Frequency)

Asia_new2<- full_join(China %>% dplyr::select(Allele, Country), `Sri Lanka`%>% dplyr::select(Allele, Country), by=c('Allele')) %>%
  full_join(`Hong Kong`%>% dplyr::select(Allele, Country), by=c('Allele')) %>%
  full_join(India%>% dplyr::select(Allele, Country), by=c('Allele')) %>%
  full_join(Iran%>% dplyr::select(Allele, Country), by=c('Allele')) %>%
  full_join(Israel%>% dplyr::select(Allele, Country), by=c('Allele'))%>%
  full_join(`Saudi Arabia`%>% dplyr::select(Allele, Country), by=c('Allele')) %>%
  full_join(Malaysia%>% dplyr::select(Allele, Country), by=c('Allele'))%>%
  full_join(Pakistan%>% dplyr::select(Allele, Country), by=c('Allele')) %>%
  full_join(`South Korea`%>% dplyr::select(Allele, Country), by=c('Allele')) %>%
  full_join(Taiwan%>% dplyr::select(Allele, Country), by=c('Allele'))%>%
  full_join(Thailand%>% dplyr::select(Allele, Country), by=c('Allele')) %>%
  full_join(Turkey%>% dplyr::select(Allele, Country), by=c('Allele')) %>% arrange(Allele)

Asia_new2<- Asia_new2%>%
  mutate(INFO= paste0( 'Count: ',  ifelse( !is.na(Country.x), 1, 0)+ifelse( !is.na(Country.y), 1, 0)+
                         ifelse( !is.na(Country.x.x), 1, 0)+ifelse( !is.na(Country.y.y), 1, 0)+
                         ifelse( !is.na(Country.x.x.x), 1, 0)+ifelse( !is.na(Country.y.y.y), 1, 0)+
                         ifelse( !is.na(Country.x.x.x.x), 1, 0)+ifelse( !is.na(Country.y.y.y.y), 1, 0)+
                         ifelse( !is.na(Country.x.x.x.x.x), 1, 0)+ifelse( !is.na(Country.y.y.y.y.y), 1, 0)+
                         ifelse( !is.na(Country.x.x.x.x.x.x), 1, 0)+ifelse( !is.na(Country.y.y.y.y.y.y), 1, 0)+
                         ifelse( !is.na(Country), 1, 0)
                       ,'/',length(Asia), ', Country: ', 
                       ifelse( !is.na(Country.x), 'China ', ''),ifelse( !is.na(Country.y), 'Sri Lanka ', ''),
                       ifelse( !is.na(Country.x.x), 'Hong Kong ', ''), ifelse( !is.na(Country.y.y), 'India ', ''), 
                       ifelse( !is.na(Country.x.x.x), 'Iran ', ''),ifelse( !is.na(Country.y.y.y), 'Israel ', ''),
                       ifelse( !is.na(Country.x.x.x.x), 'Saudi Arabia ', ''), ifelse( !is.na(Country.y.y.y.y), 'Malaysia ', ''), 
                       ifelse( !is.na(Country.x.x.x.x.x), 'Pakistan ', ''),ifelse( !is.na(Country.y.y.y.y.y), 'South Korea ', ''),
                       ifelse( !is.na(Country.x.x.x.x.x.x), 'Taiwan ', ''), ifelse( !is.na(Country.y.y.y.y.y.y), 'Thailand ', ''),
                       ifelse( !is.na(Country), 'Turkey ', '')))%>% 
  dplyr::select(Allele, INFO)


Asia_tot<-dplyr::full_join(Asia_new1 , Asia_new2, by=c('Allele') ) %>% 
  dplyr::select(Country, Geo, Allele, Frequency, INFO)

colnames(Asia_tot)<- c('Country', 'Geo', 'HLA_type', 'Frequency', 'INFO')





for (i in 1:length(Europe)) {
  print(i)
  assign(Europe[i] ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                               sheet = Europe[i], # sheet name to read from
                               skip = 2,
                               col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample',	'Location'), # TRUE to use the first row as column names
                               col_types = "guess", # guess the types of columns
                               na = "NA" )%>%  dplyr::filter((Frequency>=0.01)&(Sample>=50))%>%
           dplyr::select(-Line,-individuals,-Population, -Location)%>% 
           mutate(Country=Europe[i],
                  Type=ifelse(substr(Allele,1,1)=='A', 'A',
                              ifelse(substr(Allele,1,1)=='B', 'B',
                                     ifelse(substr(Allele,1,1)=='C', 'C',
                                            ifelse(substr(Allele,1,2)=='DQ', 'DQ',
                                                   ifelse(substr(Allele,1,2)=='DP', 'DP',
                                                          ifelse(substr(Allele,1,2)=='DR', 'DR',substr(Allele,1,2))
                                                   )
                                            )
                                     )
                              )
                  ),
                  Geo='Europe')%>% dplyr::filter(Type %in% c('A', 'B', 'C'))%>% dplyr::select(-Sample, -Type) %>% dplyr::arrange(Allele, Frequency)) 
}

# Serbia는 location이 존재하지 않음.
assign("Serbia" ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                            sheet = "Serbia", # sheet name to read from
                            skip = 2,
                            col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample'), # TRUE to use the first row as column names
                            col_types = "guess", # guess the types of columns
                            na = "NA" )%>%  dplyr::filter((Frequency>=0.01)&(Sample>=50))%>%
         dplyr::select(-Line,-individuals,-Population)%>% 
         mutate(Country="Serbia",
                Type=ifelse(substr(Allele,1,1)=='A', 'A',
                            ifelse(substr(Allele,1,1)=='B', 'B',
                                   ifelse(substr(Allele,1,1)=='C', 'C',
                                          ifelse(substr(Allele,1,2)=='DQ', 'DQ',
                                                 ifelse(substr(Allele,1,2)=='DP', 'DP',
                                                        ifelse(substr(Allele,1,2)=='DR', 'DR',substr(Allele,1,2))
                                                 )
                                          )
                                   )
                            )
                ),
                Geo='Europe')%>% dplyr::filter(Type %in% c('A', 'B', 'C'))%>% dplyr::select(-Sample, -Type) %>% dplyr::arrange(Allele, Frequency))


Russia = Russia[!duplicated(Russia[,c('Allele')]),] 
Finland = Finland[!duplicated(Finland[,c('Allele')]),] 
`Czech Republic` = `Czech Republic`[!duplicated(`Czech Republic`[,c('Allele')]),] 
France = France[!duplicated(France[,c('Allele')]),] 
Germany = Germany[!duplicated(Germany[,c('Allele')]),] 
Greece = Greece[!duplicated(Greece[,c('Allele')]),] 
Italy = Italy[!duplicated(Italy[,c('Allele')]),] 
Netherlands = Netherlands[!duplicated(Netherlands[,c('Allele')]),] 
Poland = Poland[!duplicated(Poland[,c('Allele')]),] 
Serbia = Serbia[!duplicated(Serbia[,c('Allele')]),] 
Spain = Spain[!duplicated(Spain[,c('Allele')]),] 
Sweden = Sweden[!duplicated(Sweden[,c('Allele')]),] 


Europe_new1= bind_rows(Russia, Finland, `Czech Republic`, France, Germany,Greece, Italy, Netherlands,
                       Poland, Serbia, Spain,  Sweden ) %>% 
  dplyr::select( Allele, Country, Geo, Frequency)


Europe_new2<- full_join(Russia %>% dplyr::select(Allele, Country), Finland%>% dplyr::select(Allele, Country), by=c('Allele')) %>%
  full_join(`Czech Republic`%>% dplyr::select(Allele, Country), by=c('Allele')) %>%
  full_join(France%>% dplyr::select(Allele, Country), by=c('Allele')) %>%
  full_join(Germany%>% dplyr::select(Allele, Country), by=c('Allele')) %>%
  full_join(Greece%>% dplyr::select(Allele, Country), by=c('Allele'))%>%
  full_join(Italy%>% dplyr::select(Allele, Country), by=c('Allele')) %>%
  full_join(Netherlands%>% dplyr::select(Allele, Country), by=c('Allele'))%>%
  full_join(Poland%>% dplyr::select(Allele, Country), by=c('Allele')) %>%
  full_join(Serbia%>% dplyr::select(Allele, Country), by=c('Allele')) %>%
  full_join(Spain%>% dplyr::select(Allele, Country), by=c('Allele'))%>%
  full_join(Sweden%>% dplyr::select(Allele, Country), by=c('Allele')) %>% arrange(Allele) 

  
  Europe_new2<-Europe_new2 %>%   mutate(INFO= paste0( 'Count: ',  ifelse( !is.na(Country.x), 1, 0)+ifelse( !is.na(Country.y), 1, 0)+
                         ifelse( !is.na(Country.x.x), 1, 0)+ifelse( !is.na(Country.y.y), 1, 0)+
                         ifelse( !is.na(Country.x.x.x), 1, 0)+ifelse( !is.na(Country.y.y.y), 1, 0)+
                         ifelse( !is.na(Country.x.x.x.x), 1, 0)+ifelse( !is.na(Country.y.y.y.y), 1, 0)+
                         ifelse( !is.na(Country.x.x.x.x.x), 1, 0)+ifelse( !is.na(Country.y.y.y.y.y), 1, 0)+
                         ifelse( !is.na(Country.x.x.x.x.x.x), 1, 0)+ifelse( !is.na(Country.y.y.y.y.y.y), 1, 0)
                       ,'/',length(Europe), ', Country: ', 
                       ifelse( !is.na(Country.x), 'Russia ', ''),ifelse( !is.na(Country.y), 'Finland ', ''),
                       ifelse( !is.na(Country.x.x), 'Czech Republic ', ''), ifelse( !is.na(Country.y.y), 'France ', ''), 
                       ifelse( !is.na(Country.x.x.x), 'Germany ', ''),ifelse( !is.na(Country.y.y.y), 'Greece ', ''),
                       ifelse( !is.na(Country.x.x.x.x), 'Italy ', ''), ifelse( !is.na(Country.y.y.y.y), 'Netherlands ', ''), 
                       ifelse( !is.na(Country.x.x.x.x.x), 'Poland ', ''),ifelse( !is.na(Country.y.y.y.y.y), 'Serbia ', ''),
                       ifelse( !is.na(Country.x.x.x.x.x.x), 'Spain ', ''), ifelse( !is.na(Country.y.y.y.y.y), 'Sweden ', '')))%>% 
  dplyr::select(Allele, INFO)


Europe_tot<-dplyr::full_join(Europe_new1 , Europe_new2, by=c('Allele') ) %>% 
  dplyr::select(Country, Geo, Allele, Frequency, INFO)

colnames(Europe_tot)<- c('Country', 'Geo', 'HLA_type', 'Frequency', 'INFO')




for (i in 1:length(NorthAmerica)) {
  print(i)
  assign(NorthAmerica[i] ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                                     sheet = NorthAmerica[i], # sheet name to read from
                                     skip = 2,
                                     col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample',	'Location'), # TRUE to use the first row as column names
                                     col_types = "guess", # guess the types of columns
                                     na = "NA" )%>%  dplyr::filter((Frequency>=0.01)&(Sample>=50))%>%
           dplyr::select(-Line,-individuals,-Population, -Location)%>% 
           mutate(Country=NorthAmerica[i],
                  Type=ifelse(substr(Allele,1,1)=='A', 'A',
                              ifelse(substr(Allele,1,1)=='B', 'B',
                                     ifelse(substr(Allele,1,1)=='C', 'C',
                                            ifelse(substr(Allele,1,2)=='DQ', 'DQ',
                                                   ifelse(substr(Allele,1,2)=='DP', 'DP',
                                                          ifelse(substr(Allele,1,2)=='DR', 'DR',substr(Allele,1,2))
                                                   )
                                            )
                                     )
                              )
                  ),
                  Geo='NorthAmerica')%>% dplyr::filter(Type %in% c('A', 'B', 'C'))%>% dplyr::select(-Sample, -Type) %>% dplyr::arrange(Allele, Frequency)) 
}

Jamaica = Jamaica[!duplicated(Jamaica[,c('Allele')]),] 
USA = USA[!duplicated(USA[,c('Allele')]),] 

NorthAmerica_new1= bind_rows(Jamaica, USA ) %>% 
  dplyr::select( Allele, Country, Geo, Frequency)

NorthAmerica_new2<- full_join(USA %>% dplyr::select(Allele, Country), Jamaica%>% dplyr::select(Allele, Country), by=c('Allele')) %>% arrange(Allele) %>%
  mutate(INFO= paste0( 'Count: ',  ifelse( !is.na(Country.x), 1, 0)+ifelse( !is.na(Country.y), 1, 0)
                       ,'/',length(NorthAmerica)-1, ', Country: ', 
                       ifelse( !is.na(Country.x), 'USA ', ''),ifelse( !is.na(Country.y), 'Jamica ', '')))%>% 
  dplyr::select(Allele, INFO)


NorthAmerica_tot<-dplyr::full_join(NorthAmerica_new1 , NorthAmerica_new2, by=c('Allele') ) %>% 
  dplyr::select(Country, Geo, Allele, Frequency, INFO)

colnames(NorthAmerica_tot)<- c('Country', 'Geo', 'HLA_type', 'Frequency', 'INFO')




for (i in 1:length(Oceania)) {
  print(i)
  assign(Oceania[i] ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                                sheet = Oceania[i], # sheet name to read from
                                skip = 2,
                                col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample',	'Location'), # TRUE to use the first row as column names
                                col_types = "guess", # guess the types of columns
                                na = "NA" )%>%  dplyr::filter((Frequency>=0.01)&(Sample>=50))%>%
           dplyr::select(-Line,-individuals,-Population, -Location)%>% 
           mutate(Country=Oceania[i],
                  Type=ifelse(substr(Allele,1,1)=='A', 'A',
                              ifelse(substr(Allele,1,1)=='B', 'B',
                                     ifelse(substr(Allele,1,1)=='C', 'C',
                                            ifelse(substr(Allele,1,2)=='DQ', 'DQ',
                                                   ifelse(substr(Allele,1,2)=='DP', 'DP',
                                                          ifelse(substr(Allele,1,2)=='DR', 'DR',substr(Allele,1,2))
                                                   )
                                            )
                                     )
                              )
                  ),
                  Geo='Oceania')%>% dplyr::filter(Type %in% c('A', 'B', 'C'))%>% dplyr::select(-Sample, -Type) %>% dplyr::arrange(Allele, Frequency)) 
}

Australia = Australia[!duplicated(Australia[,c('Allele')]),] 


Oceania_new1= Australia %>% 
  dplyr::select( Allele, Country, Geo, Frequency) 

Oceania_new2= Australia %>% dplyr::select(Allele, Country) %>% arrange(Allele) %>%
  mutate(INFO= paste0( 'Count: ',  ifelse( !is.na(Country), 1, 0),'/',length(Oceania), ', Country: ', 
                       ifelse( !is.na(Country), 'Australia ', '')))%>% 
  dplyr::select(Allele, INFO)



Oceania_tot<-dplyr::full_join(Oceania_new1 , Oceania_new2, by=c('Allele') ) %>% 
  dplyr::select(Country, Geo, Allele, Frequency, INFO)

colnames(Oceania_tot)<- c('Country', 'Geo', 'HLA_type', 'Frequency', 'INFO')



for (i in 1:length(SouthAmerica)) {
  print(i)
  assign(SouthAmerica[i] ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                                     sheet = SouthAmerica[i], # sheet name to read from
                                     skip = 2,
                                     col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample',	'Location'), # TRUE to use the first row as column names
                                     col_types = "guess", # guess the types of columns
                                     na = "NA" )%>%  dplyr::filter((Frequency>=0.01)&(Sample>=50))%>%
           dplyr::select(-Line,-individuals,-Population, -Location)%>% 
           mutate(Country=SouthAmerica[i],
                  Type=ifelse(substr(Allele,1,1)=='A', 'A',
                              ifelse(substr(Allele,1,1)=='B', 'B',
                                     ifelse(substr(Allele,1,1)=='C', 'C',
                                            ifelse(substr(Allele,1,2)=='DQ', 'DQ',
                                                   ifelse(substr(Allele,1,2)=='DP', 'DP',
                                                          ifelse(substr(Allele,1,2)=='DR', 'DR',substr(Allele,1,2))
                                                   )
                                            )
                                     )
                              )
                  ),
                  Geo='SouthAmerica')%>% dplyr::filter(Type %in% c('A', 'B', 'C'))%>% dplyr::select(-Sample, -Type) %>% dplyr::arrange(Allele, Frequency)) 
}

Brazil = Brazil[!duplicated(Brazil[,c('Allele')]),] 
Colombia = Colombia[!duplicated(Colombia[,c('Allele')]),] 
Peru = Peru[!duplicated(Peru[,c('Allele')]),] 


SouthAmerica_new1= bind_rows(Brazil, Colombia, Peru ) %>% 
  dplyr::select( Allele, Country, Geo, Frequency) %>% mutate(count=1)



SouthAmerica_new2<- full_join(Brazil %>% dplyr::select(Allele, Country), Colombia%>% dplyr::select(Allele, Country), by=c('Allele')) %>%
  full_join(Peru%>% dplyr::select(Allele, Country), by=c('Allele')) %>%arrange(Allele)

SouthAmerica_new2<-SouthAmerica_new2 %>%
  mutate(INFO= paste0( 'Count: ',  ifelse( !is.na(Country.x), 1, 0)+ifelse( !is.na(Country.y), 1, 0)+
                         ifelse( !is.na(Country), 1, 0)
                       ,'/',length(SouthAmerica), ', Country: ', 
                       ifelse( !is.na(Country.x), 'Brazil ', ''),ifelse( !is.na(Country.y), 'Colombia ', ''),
                       ifelse( !is.na(Country), 'Peru ', '')))%>% 
  dplyr::select(Allele, INFO)


SouthAmerica_tot<-dplyr::full_join(SouthAmerica_new1 , SouthAmerica_new2, by=c('Allele') ) %>% 
  dplyr::select(Country, Geo, Allele, Frequency, INFO)

colnames(SouthAmerica_tot)<- c('Country', 'Geo', 'HLA_type', 'Frequency', 'INFO')




result <-bind_rows(Africa_tot, Asia_tot, Europe_tot, NorthAmerica_tot, Oceania_tot, SouthAmerica_tot)


Africa_tot = Africa_tot[!duplicated(Africa_tot[,c('HLA_type', 'Geo')]),] %>% dplyr::select(HLA_type, Geo, INFO)
Asia_tot = Asia_tot[!duplicated(Asia_tot[,c('HLA_type', 'Geo')]),] %>% dplyr::select(HLA_type, Geo, INFO)
Europe_tot = Europe_tot[!duplicated(Europe_tot[,c('HLA_type', 'Geo')]),] %>% dplyr::select(HLA_type, Geo, INFO)
NorthAmerica_tot = NorthAmerica_tot[!duplicated(NorthAmerica_tot[,c('HLA_type', 'Geo')]),] %>% dplyr::select(HLA_type, Geo, INFO)
Oceania_tot = Oceania_tot[!duplicated(Oceania_tot[,c('HLA_type', 'Geo')]),] %>% dplyr::select(HLA_type, Geo, INFO)
SouthAmerica_tot = SouthAmerica_tot[!duplicated(SouthAmerica_tot[,c('HLA_type', 'Geo')]),] %>% dplyr::select(HLA_type, Geo, INFO)


result_geo <- full_join(Africa_tot , Asia_tot, by=c('HLA_type')) %>%
  full_join(Europe_tot, by=c('HLA_type')) %>%
  full_join(NorthAmerica_tot, by=c('HLA_type')) %>%
  full_join(Oceania_tot, by=c('HLA_type')) %>%
  full_join(SouthAmerica_tot, by=c('HLA_type')) %>%arrange(HLA_type)

  result_geo <- result_geo %>%  mutate(Geo_INFO= paste0( 'Count: ',
                           ifelse( !is.na(result_geo$INFO.x), substr(result_geo$INFO.x, 7,8)%>% as.numeric() , 0)+
                           ifelse( !is.na(result_geo$INFO.y), gsub("/","" ,substr(result_geo$INFO.y, 7,9))%>% as.numeric() , 0)+
                           ifelse( !is.na(result_geo$INFO.x.x), gsub("/","" ,substr(result_geo$INFO.x.x, 7,9))%>% as.numeric() , 0)+
                           ifelse( !is.na(result_geo$INFO.y.y), substr(result_geo$INFO.y.y, 7,8)%>% as.numeric() , 0)+
                           ifelse( !is.na(result_geo$INFO.x.x.x), substr(result_geo$INFO.x.x.x, 7,8)%>% as.numeric() , 0)+
                           ifelse( !is.na(result_geo$INFO.y.y.y), substr(result_geo$INFO.y.y.y, 7,8)%>% as.numeric() , 0),
                           '/',length(c(Africa , Asia, Europe, NorthAmerica, Oceania, SouthAmerica))-2, 
                           ', Africa: ',ifelse(is.na(result_geo$INFO.x), 0,substr(result_geo$INFO.x, 7,8)),
                           ' ', ifelse(is.na(result_geo$INFO.x),'',substr(result_geo$INFO.x, 22,300)),
                           
                           ', Asia: ', ifelse(is.na(result_geo$INFO.y), 0,gsub("/","" ,substr(result_geo$INFO.y, 7,9))),
                           ' ', ifelse(is.na(result_geo$INFO.y),'',gsub(":","" ,substr(result_geo$INFO.y, 23,300))),
                           
                           ', Europe: ', ifelse(is.na(result_geo$INFO.x.x), 0,gsub("/","" ,substr(result_geo$INFO.x.x, 7,9))),
                           ' ', ifelse(is.na(result_geo$INFO.x.x),'',gsub(":","" ,substr(result_geo$INFO.x.x, 23,300))),
                           
                           ', NorthAmerica: ', ifelse(is.na(result_geo$INFO.y.y), 0,substr(result_geo$INFO.y.y, 7,8)),
                           ' ', ifelse(is.na(result_geo$INFO.y.y),'',substr(result_geo$INFO.y.y, 22,300)),
                           
                           ', Oceania: ', ifelse(is.na(result_geo$INFO.x.x.x), 0,substr(result_geo$INFO.x.x.x, 7,8)),
                           ' ', ifelse(is.na(result_geo$INFO.x.x.x),'',substr(result_geo$INFO.x.x.x, 22,300)),
                           
                           ', SouthAmerica: ', ifelse(is.na(result_geo$INFO.y.y.y), 0,substr(result_geo$INFO.y.y.y, 7,8)),
                           ' ', ifelse(is.na(result_geo$INFO.y.y.y),'',substr(result_geo$INFO.y.y.y, 22,300))))%>% 
  dplyr::select(HLA_type, Geo_INFO)


  result_total <- full_join(result , result_geo, by=c('HLA_type')) %>%
    arrange(HLA_type)
  
  
  



write.table(result_total, "C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/output3.txt",
            sep = "\t",
            row.names = F)

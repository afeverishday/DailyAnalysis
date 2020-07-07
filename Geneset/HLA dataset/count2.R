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
                  Geo='Africa')%>% dplyr::filter(Type %in% c('A', 'B', 'C')) %>% dplyr::arrange(Allele, Frequency)) 
}

Morocco = Morocco[!duplicated(Morocco[,c('Allele')]),] 
Kenya = Kenya[!duplicated(Kenya[,c('Allele')]),] 
Nigeria = Nigeria[!duplicated(Nigeria[,c('Allele')]),] 
`South Africa`= `South Africa`[!duplicated(`South Africa`[,c('Allele')]),] 
Tunisia = Tunisia[!duplicated(Tunisia[,c('Allele')]),] 



Africa_new1= bind_rows(Morocco, Kenya, Nigeria, `South Africa`, Tunisia) %>% 
  dplyr::select(-Sample, -Type, Allele, Country, Geo,Frequency) %>% mutate(count=1)

Africa_new2= Africa_new1 %>% spread( key='Country', value='count')

Africa_new2[is.na(Africa_new2)]<-0

library(tidyr)
Africa_new2<-Africa_new2 %>%  
  mutate(INFO= paste0( 'Count: ',Kenya+Morocco+`South Africa`+Tunisia,'/',length(Africa)-1, ', Country: ', 
                       ifelse( Kenya==1, 'Kenya ', ''),ifelse(Morocco==1, 'Morocco ', ''),
                       ifelse( Nigeria==1, 'Nigeria ', ''),ifelse(`South Africa`==1, 'South Africa ', ''),
                       ifelse( Tunisia==1, 'Tunisia ', ''))) %>% 
  dplyr::select(-Kenya , -Morocco, -`South Africa`,-Tunisia, -Geo)

Africa_tot<-dplyr::full_join(Africa_new1 , Africa_new2, by=c('Allele','Frequency') ) %>% 
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
                  Geo='Asia')%>% dplyr::filter(Type %in% c('A', 'B', 'C')) %>% dplyr::arrange(Allele, Frequency)) 
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


Asia_new1= bind_rows(China,`Sri Lanka`,`Hong Kong`,India,Iran, Israel, `Saudi Arabia`, Malaysia,
                     Pakistan,`South Korea`, Taiwan, Thailand, Turkey ) %>% 
  dplyr::select(-Sample, -Type, Allele, Country, Geo, Frequency) %>% mutate(count=1)

Asia_new2= Asia_new1 %>% spread( key='Country', value='count')

Asia_new2[is.na(Asia_new2)]<-0

library(tidyr)
Asia_new2<-Asia_new2 %>%  
  mutate(INFO= paste0( 'Count: ',China+`Sri Lanka`+`Hong Kong`+India+Iran+ Israel+ `Saudi Arabia`+ Malaysia+ Pakistan+`South Korea`+ Taiwan+ Thailand+ Turkey,'/',length(Asia), 
                       ', Country: ', ifelse( China==1, 'China ', ''),ifelse(`Sri Lanka`==1, 'Sri Lanka ', ''),
                       ifelse( `Hong Kong`==1, 'Hong Kong ', ''), ifelse(India==1, 'India ', ''), 
                       ifelse( Iran==1, 'Iran ', ''), ifelse( Israel==1, 'Israel ', ''),
                       ifelse( `Saudi Arabia`==1, 'Saudi Arabia ', ''), ifelse( Malaysia==1, 'Malaysia ', ''),
                       ifelse( Pakistan==1, 'Pakistan ', ''),ifelse( `South Korea`==1, 'South Korea ', ''),
                       ifelse( Taiwan==1, 'Taiwan ', ''),ifelse( Thailand==1, 'Thailand ', ''),ifelse( Turkey==1, 'Turkey ', ''))) %>% 
  dplyr::select(-China, -`Sri Lanka`,-`Hong Kong`,-India,-Iran,-Israel,-`Saudi Arabia`,-Malaysia,-Pakistan,-`South Korea`,
                -Taiwan,-Thailand,-Turkey,  -Geo)

Asia_tot<-dplyr::full_join(Asia_new1 , Asia_new2, by=c('Allele','Frequency') ) %>% 
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
                  Geo='Europe')%>% dplyr::filter(Type %in% c('A', 'B', 'C')) %>% dplyr::arrange(Allele, Frequency)) 
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
                Geo='Europe')%>% dplyr::filter(Type %in% c('A', 'B', 'C')) %>% dplyr::arrange(Allele, Frequency))


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
                       Poland, Spain,  Sweden ) %>% 
  dplyr::select(-Sample, -Type, Allele, Country, Geo, Frequency) %>% mutate(count=1)

Europe_new2= Europe_new1 %>% spread( key='Country', value='count')

Europe_new2[is.na(Europe_new2)]<-0

library(tidyr)
Europe_new2<-Europe_new2 %>%  
  mutate(INFO= paste0( 'Count: ',Russia+ Finland+ `Czech Republic`+ France+ Germany+Greece+ Italy+ Netherlands+  Poland+ Spain+  Sweden,'/',length(Europe), 
                       ', Country: ', ifelse( Russia==1, 'Russia ', ''),ifelse(Finland==1, 'Finland ', ''),
                       ifelse( `Czech Republic`==1, 'Czech Republic ', ''), ifelse(France==1, 'France ', ''), 
                       ifelse( Germany==1, 'Germany ', ''), ifelse( Greece==1, 'Greece ', ''),
                       ifelse( Italy==1, 'Italy ', ''), ifelse( Netherlands==1, 'Netherlands ', ''),
                       ifelse( Poland==1, 'Poland ', ''),ifelse( Spain==1, 'Spain ', ''),
                       ifelse( Sweden==1, 'Sweden ', ''))) %>% 
  dplyr::select(-Russia, -Finland, -`Czech Republic`, -France, -Germany, -Greece, -Italy, -Netherlands, -Poland, -Spain, -Sweden,  -Geo)

Europe_tot<-dplyr::full_join(Europe_new1 , Europe_new2, by=c('Allele','Frequency') ) %>% 
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
                  Geo='NorthAmerica')%>% dplyr::filter(Type %in% c('A', 'B', 'C')) %>% dplyr::arrange(Allele, Frequency)) 
}

Jamaica = Jamaica[!duplicated(Jamaica[,c('Allele')]),] 
USA = USA[!duplicated(USA[,c('Allele')]),] 

NorthAmerica_new1= bind_rows(Jamaica, USA ) %>% 
  dplyr::select(-Sample, -Type, Allele, Country, Geo, Frequency) %>% mutate(count=1)

NorthAmerica_new2= NorthAmerica_new1 %>% spread( key='Country', value='count')

NorthAmerica_new2[is.na(NorthAmerica_new2)]<-0

library(tidyr)
NorthAmerica_new2<-NorthAmerica_new2 %>%  
  mutate(INFO= paste0( 'Count: ', USA,'/',length(NorthAmerica)-1, 
                       ', Country: ', ifelse( Jamaica==1, 'Jamaica ', ''),ifelse(USA==1, 'USA ', ''))) %>% 
  dplyr::select( -USA,  -Geo)

NorthAmerica_tot<-dplyr::full_join(NorthAmerica_new1 , NorthAmerica_new2, by=c('Allele','Frequency') ) %>% 
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
                  Geo='Oceania')%>% dplyr::filter(Type %in% c('A', 'B', 'C')) %>% dplyr::arrange(Allele, Frequency)) 
}

Australia = Australia[!duplicated(Australia[,c('Allele')]),] 


Oceania_new1= Australia %>% 
  dplyr::select(-Sample, -Type, Allele, Country, Geo, Frequency) %>% mutate(count=1)

Oceania_new2= Oceania_new1 %>% spread( key='Country', value='count')

Oceania_new2[is.na(Oceania_new2)]<-0

library(tidyr)
Oceania_new2<-Oceania_new2 %>%  
  mutate(INFO= paste0( 'Count: ', Australia,'/',length(Oceania), 
                       ', Country: ',ifelse(Australia==1, 'Australia ', ''))) %>% 
  dplyr::select( -Australia,  -Geo)

Oceania_tot<-dplyr::full_join(Oceania_new1 , Oceania_new2, by=c('Allele','Frequency') ) %>% 
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
                  Geo='SouthAmerica')%>% dplyr::filter(Type %in% c('A', 'B', 'C')) %>% dplyr::arrange(Allele, Frequency)) 
}

Brazil = Brazil[!duplicated(Brazil[,c('Allele')]),] 
Colombia = Colombia[!duplicated(Colombia[,c('Allele')]),] 
Peru = Peru[!duplicated(Peru[,c('Allele')]),] 


SouthAmerica_new1= bind_rows(Brazil, Colombia, Peru ) %>% 
  dplyr::select(-Sample, -Type, Allele, Country, Geo, Frequency) %>% mutate(count=1)

SouthAmerica_new2= SouthAmerica_new1 %>% spread( key='Country', value='count')

SouthAmerica_new2[is.na(SouthAmerica_new2)]<-0

library(tidyr)
SouthAmerica_new2<-SouthAmerica_new2 %>%  
  mutate(INFO= paste0( 'Count: ', Brazil+ Colombia+ Peru,'/',length(SouthAmerica), 
                       ', Country: ', ifelse( Brazil==1, 'Brazil ', ''), ifelse(Colombia==1, 'Colombia ', ''), 
                       ifelse(Peru==1, 'Peru ', ''))) %>% 
  dplyr::select( -Brazil,-Colombia, -Peru,  -Geo)

SouthAmerica_tot<-dplyr::full_join(SouthAmerica_new1 , SouthAmerica_new2, by=c('Allele','Frequency') ) %>% 
  dplyr::select(Country, Geo, Allele, Frequency, INFO)

colnames(SouthAmerica_tot)<- c('Country', 'Geo', 'HLA_type', 'Frequency', 'INFO')




result <-bind_rows(Africa_tot, Asia_tot, Europe_tot, NorthAmerica_tot, Oceania_tot, SouthAmerica_tot)


write.table(result, "C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/output3.txt",
            sep = "\t",
            row.names = F)

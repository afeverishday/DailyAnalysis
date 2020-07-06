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
           mutate(country=Africa[i],
                  type=ifelse(substr(Allele,1,1)=='A', 'A',
                              ifelse(substr(Allele,1,1)=='B', 'B',
                                     ifelse(substr(Allele,1,1)=='C', 'C',
                                            ifelse(substr(Allele,1,2)=='DQ', 'DQ',
                                                   ifelse(substr(Allele,1,2)=='DP', 'DP',
                                                          ifelse(substr(Allele,1,2)=='DR', 'DR',substr(Allele,1,2))
                                                          )
                                                   )
                                            )
                                     )
                              )) %>% dplyr::arrange(Allele, Frequency)) 
  }

Morocco = Morocco[!duplicated(Morocco[,c('Allele')]),] 
Kenya = Kenya[!duplicated(Kenya[,c('Allele')]),] 
Nigeria = Nigeria[!duplicated(Nigeria[,c('Allele')]),] 
`South Africa`= `South Africa`[!duplicated(`South Africa`[,c('Allele')]),] 
Tunisia = Tunisia[!duplicated(Tunisia[,c('Allele')]),] 



Africa_tot<- full_join(Morocco, Kenya, by=c('Allele','type')) %>%full_join(Nigeria, by=c('Allele','type')) %>%
  full_join(`South Africa`, by=c('Allele','type')) %>%full_join(Tunisia, by=c('Allele','type')) %>%
  mutate(Geo="Africa",
         Tot_Count=length(Africa),
         Count= (ifelse(is.na(Frequency.x),0,1) +ifelse(is.na(Frequency.y),0,1)+
           ifelse(is.na(Frequency.x.x),0,1)+ifelse(is.na(Frequency.y.y),0,1)+ifelse(is.na(Frequency),0,1)) ,
         INFO= paste0(  "Morocco: " , Frequency.x ,", ", "Kenya: " , Frequency.y ,", ",
            "Nigeria: " , Frequency.x.x ,", " , "South Africa: " , Frequency.y.y ,", ",
            "Tunisia: " , Frequency )) %>%  select(Geo, Allele, Count, Tot_Count, INFO)




for (i in 1:length(Asia)) {
  print(i)
  assign(Asia[i] ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                               sheet = Asia[i], # sheet name to read from
                               skip = 2,
                               col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample',	'Location'), # TRUE to use the first row as column names
                               col_types = "guess", # guess the types of columns
                               na = "NA" )%>%  dplyr::filter((Frequency>=0.01)&(Sample>=50))%>%
           dplyr::select(-Line,-individuals,-Population, -Location)%>% 
           mutate(country=Asia[i],
                  type=ifelse(substr(Allele,1,1)=='A', 'A',
                              ifelse(substr(Allele,1,1)=='B', 'B',
                                     ifelse(substr(Allele,1,1)=='C', 'C',
                                            ifelse(substr(Allele,1,2)=='DQ', 'DQ',
                                                   ifelse(substr(Allele,1,2)=='DP', 'DP',
                                                          ifelse(substr(Allele,1,2)=='DR', 'DR',substr(Allele,1,2))
                                                   )
                                            )
                                     )
                              )
                  )) %>% dplyr::arrange(Allele, Frequency)) 
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



Asia_tot<- full_join(China, `Sri Lanka`, by=c('Allele','type')) %>%full_join(`Hong Kong`, by=c('Allele','type')) %>%
  full_join(India, by=c('Allele','type')) %>% full_join(Iran, by=c('Allele','type')) %>% 
  full_join(Israel, by=c('Allele','type')) %>% full_join(`Saudi Arabia`, by=c('Allele','type')) %>%
  full_join(Malaysia, by=c('Allele','type')) %>% full_join(Pakistan, by=c('Allele','type')) %>% 
  full_join(`South Korea`, by=c('Allele','type')) %>% full_join(Taiwan, by=c('Allele','type')) %>% 
  full_join(Thailand, by=c('Allele','type')) %>% full_join(Turkey, by=c('Allele','type')) %>%
  mutate(Geo="Asia",
         Tot_Count=length(Asia),
         Count= (ifelse(is.na(Frequency.x),0,1) +ifelse(is.na(Frequency.y),0,1)+
                   ifelse(is.na(Frequency.x.x),0,1)+ifelse(is.na(Frequency.y.y),0,1)+ifelse(is.na(Frequency.x.x.x),0,1)+
                   ifelse(is.na(Frequency.y.y.y),0,1)+ifelse(is.na(Frequency.x.x.x.x),0,1)+ifelse(is.na(Frequency.y.y.y.y),0,1)+
                   ifelse(is.na(Frequency.x.x.x.x.x),0,1)+ifelse(is.na(Frequency.y.y.y.y.y),0,1)+ifelse(is.na(Frequency.x.x.x.x.x.x),0,1)+
                   ifelse(is.na(Frequency.y.y.y.y.y.y),0,1)+ifelse(is.na(Frequency),0,1)) ,
         INFO= paste0( "China: " , Frequency.x ,", ", "Sri Lanka: " , Frequency.y ,", ",
                       "Hong Kong: " , Frequency.x.x ,", " , "India: " , Frequency.y.y ,", ",
                       "Iran: " , Frequency.x.x.x , "Israel: " , Frequency.y.y.y,
                       "Saudi Arabia: " , Frequency.x.x.x.x,  "Malaysia: " , Frequency.y.y.y.y,
                       "Pakistan: " , Frequency.x.x.x.x.x,  "South Korea: " , Frequency.y.y.y.y.y,
                       "Taiwan: " , Frequency.x.x.x.x.x.x, "Thailand: " , Frequency.y.y.y.y.y.y,
                       "Turkey: " , Frequency)) %>%  select(Geo, Allele, Count, Tot_Count, INFO)






for (i in 1:length(Europe)) {
  print(i)
  assign(Europe[i] ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                               sheet = Europe[i], # sheet name to read from
                               skip = 2,
                               col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample',	'Location'), # TRUE to use the first row as column names
                               col_types = "guess", # guess the types of columns
                               na = "NA" )%>%  dplyr::filter((Frequency>=0.01)&(Sample>=50))%>%
           dplyr::select(-Line,-individuals,-Population, -Location)%>% 
           mutate(country=Europe[i],
                  type=ifelse(substr(Allele,1,1)=='A', 'A',
                              ifelse(substr(Allele,1,1)=='B', 'B',
                                     ifelse(substr(Allele,1,1)=='C', 'C',
                                            ifelse(substr(Allele,1,2)=='DQ', 'DQ',
                                                   ifelse(substr(Allele,1,2)=='DP', 'DP',
                                                          ifelse(substr(Allele,1,2)=='DR', 'DR',substr(Allele,1,2))
                                                   )
                                            )
                                     )
                              )
                  )) %>% dplyr::arrange(Allele, Frequency)) 
}

# Serbia는 location이 존재하지 않음.
assign("Serbia" ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                             sheet = "Serbia", # sheet name to read from
                             skip = 2,
                             col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample'), # TRUE to use the first row as column names
                             col_types = "guess", # guess the types of columns
                             na = "NA" )%>%  dplyr::filter((Frequency>=0.01)&(Sample>=50))%>%
         dplyr::select(-Line,-individuals,-Population)%>% 
         mutate(country="Serbia",
                type=ifelse(substr(Allele,1,1)=='A', 'A',
                            ifelse(substr(Allele,1,1)=='B', 'B',
                                   ifelse(substr(Allele,1,1)=='C', 'C',
                                          ifelse(substr(Allele,1,2)=='DQ', 'DQ',
                                                 ifelse(substr(Allele,1,2)=='DP', 'DP',
                                                        ifelse(substr(Allele,1,2)=='DR', 'DR',substr(Allele,1,2))
                                                 )
                                          )
                                   )
                            )
                )) %>% dplyr::arrange(Allele, Frequency))


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



Europe_tot<- full_join(Russia, Finland, by=c('Allele','type')) %>%
  full_join(`Czech Republic`, by=c('Allele','type')) %>% full_join(France, by=c('Allele','type')) %>% 
  full_join(Germany, by=c('Allele','type')) %>% full_join(Greece, by=c('Allele','type')) %>%
  full_join(Italy, by=c('Allele','type')) %>% full_join(Netherlands, by=c('Allele','type')) %>% 
  full_join(Poland, by=c('Allele','type')) %>% full_join(Serbia, by=c('Allele','type')) %>% 
  full_join(Spain, by=c('Allele','type')) %>%full_join(Sweden, by=c('Allele','type')) %>%
  mutate(Geo="Europe",
         Tot_Count=length(Europe)+1,
         Count= (ifelse(is.na(Frequency.x),0,1) +ifelse(is.na(Frequency.y),0,1)+
                   ifelse(is.na(Frequency.x.x),0,1)+ifelse(is.na(Frequency.y.y),0,1)+ifelse(is.na(Frequency.x.x.x),0,1)+
                   ifelse(is.na(Frequency.y.y.y),0,1)+ifelse(is.na(Frequency.x.x.x.x),0,1)+ifelse(is.na(Frequency.y.y.y.y),0,1)+
                   ifelse(is.na(Frequency.x.x.x.x.x),0,1)+ifelse(is.na(Frequency.y.y.y.y.y),0,1)+ifelse(is.na(Frequency.x.x.x.x.x.x),0,1)+
                   ifelse(is.na(Frequency.y.y.y.y.y.y),0,1)) ,
         INFO= paste0( "Russia: " , Frequency.x ,", ", "Finland: " , Frequency.y ,", ",
                       "Czech Republic: " , Frequency.x.x ,", ", "France: " , Frequency.y.y ,", ",
                       "Germany: " , Frequency.x.x.x, "Greece: " , Frequency.y.y.y,
                       "Italy: " , Frequency.x.x.x.x,  "Netherlands: " , Frequency.y.y.y.y,
                       "Poland: " , Frequency.x.x.x.x.x, "Serbia: " , Frequency.y.y.y.y.y,
                       "Spain: " , Frequency.x.x.x.x.x.x,  "Sweden: " , Frequency.y.y.y.y.y.y)) %>%  select(Geo, Allele, Count, Tot_Count, INFO)





for (i in 1:length(NorthAmerica)) {
  print(i)
  assign(NorthAmerica[i] ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                                     sheet = NorthAmerica[i], # sheet name to read from
                                     skip = 2,
                                     col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample',	'Location'), # TRUE to use the first row as column names
                                     col_types = "guess", # guess the types of columns
                                     na = "NA" )%>%  dplyr::filter((Frequency>=0.01)&(Sample>=50))%>%
           dplyr::select(-Line,-individuals,-Population, -Location)%>% 
           mutate(country=NorthAmerica[i],
                  type=ifelse(substr(Allele,1,1)=='A', 'A',
                              ifelse(substr(Allele,1,1)=='B', 'B',
                                     ifelse(substr(Allele,1,1)=='C', 'C',
                                            ifelse(substr(Allele,1,2)=='DQ', 'DQ',
                                                   ifelse(substr(Allele,1,2)=='DP', 'DP',
                                                          ifelse(substr(Allele,1,2)=='DR', 'DR',substr(Allele,1,2))
                                                   )
                                            )
                                     )
                              )
                  )) %>% dplyr::arrange(Allele, Frequency)) 
}

Jamaica = Jamaica[!duplicated(Jamaica[,c('Allele')]),] 
USA = USA[!duplicated(USA[,c('Allele')]),] 


NorthAmerica_tot<- full_join(Jamaica, USA, by=c('Allele','type')) %>%
  mutate(Geo="NorthAmerica",
         Tot_Count=length(NorthAmerica),
         Count= (ifelse(is.na(Frequency.x),0,1) +ifelse(is.na(Frequency.y),0,1)) ,
         INFO= paste0(  "Jmaica: " , Frequency.x ,", ", "USA: " , Frequency.y )) %>%  select(Geo, Allele, Count, Tot_Count, INFO)



for (i in 1:length(Oceania)) {
  print(i)
  assign(Oceania[i] ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                                sheet = Oceania[i], # sheet name to read from
                                skip = 2,
                                col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample',	'Location'), # TRUE to use the first row as column names
                                col_types = "guess", # guess the types of columns
                                na = "NA" )%>%  dplyr::filter((Frequency>=0.01)&(Sample>=50))%>%
           dplyr::select(-Line,-individuals,-Population, -Location)%>% 
           mutate(country=Oceania[i],
                  type=ifelse(substr(Allele,1,1)=='A', 'A',
                              ifelse(substr(Allele,1,1)=='B', 'B',
                                     ifelse(substr(Allele,1,1)=='C', 'C',
                                            ifelse(substr(Allele,1,2)=='DQ', 'DQ',
                                                   ifelse(substr(Allele,1,2)=='DP', 'DP',
                                                          ifelse(substr(Allele,1,2)=='DR', 'DR',substr(Allele,1,2))
                                                   )
                                            )
                                     )
                              )
                  )) %>% dplyr::arrange(Allele, Frequency)) 
}

Australia = Australia[!duplicated(Australia[,c('Allele')]),] 


Oceania_tot<- Australia%>%  mutate(Geo="Oceania",
                                   Tot_Count=length(Oceania),
                                   Count= (ifelse(is.na(Frequency),0,1) ),
                                   INFO= paste0("Australia: " , Frequency )) %>%  select(Geo, Allele, Count, Tot_Count, INFO)





for (i in 1:length(SouthAmerica)) {
  print(i)
  assign(SouthAmerica[i] ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                                     sheet = SouthAmerica[i], # sheet name to read from
                                     skip = 2,
                                     col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample',	'Location'), # TRUE to use the first row as column names
                                     col_types = "guess", # guess the types of columns
                                     na = "NA" )%>%  dplyr::filter((Frequency>=0.01)&(Sample>=50))%>%
           dplyr::select(-Line,-individuals,-Population, -Location)%>% 
           mutate(country=SouthAmerica[i],
                  type=ifelse(substr(Allele,1,1)=='A', 'A',
                              ifelse(substr(Allele,1,1)=='B', 'B',
                                     ifelse(substr(Allele,1,1)=='C', 'C',
                                            ifelse(substr(Allele,1,2)=='DQ', 'DQ',
                                                   ifelse(substr(Allele,1,2)=='DP', 'DP',
                                                          ifelse(substr(Allele,1,2)=='DR', 'DR',substr(Allele,1,2))
                                                   )
                                            )
                                     )
                              )
                  )) %>% dplyr::arrange(Allele, Frequency)) 
}

Brazil = Brazil[!duplicated(Brazil[,c('Allele')]),] 
Colombia = Colombia[!duplicated(Colombia[,c('Allele')]),] 
Peru = Peru[!duplicated(Peru[,c('Allele')]),] 


SouthAmerica_tot<- full_join(Brazil, Colombia, by=c('Allele','type')) %>%full_join(Peru, by=c('Allele','type')) %>%
  mutate(Geo="SouthAmerica",
         Tot_Count=length(SouthAmerica),
         Count= (ifelse(is.na(Frequency.x),0,1) +ifelse(is.na(Frequency.y),0,1)+
                 ifelse(is.na(Frequency),0,1)) ,
         INFO= paste0(  "Brazil: " , Frequency.x ,", ", "Colombia: " , Frequency.y ,", ",
                        "Peru: " , Frequency )) %>%  select(Geo, Allele, Count, Tot_Count, INFO)



result <-bind_rows(Africa_tot, Asia_tot, Europe_tot, NorthAmerica_tot, Oceania_tot, SouthAmerica_tot)

colnames(result)<- c('Geo', 'HLA_type', 'Count', 'Tot_Count', 'INFO')

write.table(result, "C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/output1.txt",
            sep = "\t",
            row.names = F)



# 나라별 면역 type 개수 구하는 code
result_count = bind_rows(Morocco, Kenya,Nigeria,`South Africa`,Tunisia,China,`Sri Lanka`, `Hong Kong`,India,Iran,Israel, 
          `Saudi Arabia`,Malaysia,Pakistan, `South Korea`,Taiwan,Thailand,Turkey,Russia,Finland,
          `Czech Republic`,France,Germany,Greece, Italy,Netherlands,Poland,Spain,Sweden,Serbia,
          Jamaica,USA,Australia, Brazil,Colombia,Peru)%>%   dplyr::group_by(country, type) %>% dplyr::summarise(n=n()) 
library(tidyr)
result_count = result_count %>% spread(key='type', value='n')

write.table(result_count, "C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/output2.txt",
            sep = "\t",
            row.names = F)
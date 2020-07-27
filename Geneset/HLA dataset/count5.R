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
                  ncount= floor(Sample*2*Frequency)) %>%filter(ncount!=0)%>%  dplyr::arrange(Allele, desc(Frequency))) 
}

%>% dplyr::group_by(Allele) %>% dplyr::summarise(Ncount= sum(ncount),                                                                                                                Country=unique(Country))

# Frequency가 0이 아닌데 count가 0이 되는 경우
Kenya%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
`South Africa`%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()


for (i in 1:length(Asia)) {
  print(i)
  assign(Asia[i] ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                             sheet = Asia[i], # sheet name to read from
                             skip = 2,
                             col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample',	'Location'), # TRUE to use the first row as column names
                             col_types = "guess", # guess the types of columns
                             na = "NA" )%>%  dplyr::filter((Sample>=50)&(Frequency!=0)&!is.na(Frequency))%>%
           dplyr::select(-Line,-individuals,-Population, -Location)%>% 
           dplyr::filter(substr(Allele,1,1)!='D') %>%
           mutate(Country=Asia[i],
                  ncount= floor(Sample*2*Frequency))%>%filter(ncount!=0)%>%  dplyr::arrange(Allele, desc(Frequency))) 
}


# Frequency가 0이 아닌데 count가 0이 되는 경우
China%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
`Hong Kong`%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
India%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
Israel%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
Malaysia%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
Pakistan%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
`Saudi Arabia`%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
`South Korea`%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
Taiwan%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
Thailand%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
Turkey%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()



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
                  ncount= floor(Sample*2*Frequency))%>%filter(ncount!=0)%>%  dplyr::arrange(Allele, desc(Frequency))) 
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
                ncount= floor(Sample*2*Frequency))%>%filter(ncount!=0)%>%  dplyr::arrange(Allele, desc(Frequency))) 


# Frequency가 0이 아닌데 count가 0이 되는 경우
Germany%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
Greece%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
Italy%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
Netherlands%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
Russia%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
Spain%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()
Sweden%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()

for (i in 1:length(NorthAmerica)) {
  print(i)
  assign(NorthAmerica[i] ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                                     sheet = NorthAmerica[i], # sheet name to read from
                                     skip = 2,
                                     col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample',	'Location'), # TRUE to use the first row as column names
                                     col_types = "guess", # guess the types of columns
                                     na = "NA" )%>%  dplyr::filter((Sample>=50)&(Frequency!=0)&!is.na(Frequency))%>%
           dplyr::select(-Line,-individuals,-Population, -Location)%>% 
           dplyr::filter(substr(Allele,1,1)!='D') %>%
           mutate(Country=NorthAmerica[i],
                  ncount= floor(Sample*2*Frequency))%>% dplyr::arrange(Allele, desc(Frequency))%>% filter(ncount!=0)%>% 
           dplyr::group_by(Allele) %>% filter(ncount!=0)%>% dplyr::summarise(Ncount= sum(ncount),                                                                                                                Country=unique(Country))) 
}

# Frequency가 0이 아닌데 count가 0이 되는 경우
USA%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()

USA$NFrequency<- USA$Ncount/sum(USA$Ncount)

write.table(USA, "C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/USA.txt",
            sep = "\t",
            row.names = F)



for (i in 1:length(Oceania)) {
  print(i)
  assign(Oceania[i] ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                                sheet = Oceania[i], # sheet name to read from
                                skip = 2,
                                col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample',	'Location'), # TRUE to use the first row as column names
                                col_types = "guess", # guess the types of columns
                                na = "NA" )%>%  dplyr::filter((Sample>=50)&(Frequency!=0)&!is.na(Frequency))%>%
           dplyr::select(-Line,-individuals,-Population, -Location)%>% 
           dplyr::filter(substr(Allele,1,1)!='D') %>%
           mutate(Country=Oceania[i],
                  ncount= floor(Sample*2*Frequency))%>% dplyr::arrange(Allele, desc(Frequency))) 
}




for (i in 1:length(SouthAmerica)) {
  print(i)
  assign(SouthAmerica[i] ,read_excel("C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/AFND_국가별.xlsx", # path
                                     sheet = SouthAmerica[i], # sheet name to read from
                                     skip = 2,
                                     col_names = c('Line',	'Allele',	'Population',	'individuals',	'Frequency',	'Sample',	'Location'), # TRUE to use the first row as column names
                                     col_types = "guess", # guess the types of columns
                                     na = "NA" )%>%  dplyr::filter((Sample>=50)&(Frequency!=0)&!is.na(Frequency))%>%
           dplyr::select(-Line,-individuals,-Population, -Location)%>% 
           dplyr::filter(substr(Allele,1,1)!='D') %>%
           mutate(Country=SouthAmerica[i],
                  ncount= floor(Sample*2*Frequency))%>% dplyr::arrange(Allele, desc(Frequency))) 
}


# Frequency가 0이 아닌데 count가 0이 되는 경우
Brazil%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table() 
Colombia%>% filter(ncount==0) %>% group_by(Sample) %>%select(Sample) %>%table()





































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



result_total$HLA_type_new <- gsub(':','',result_total$HLA_type)


result_total<-result_total %>% mutate(
  Supertype= ifelse(HLA_type_new %in% c("A*0101", "A*0103", "A*0112", "A*2609", "A*2618", "A*3012", "A*3602", "A*2501", "A*3603", "A*2601", "A*0104", 
                                        "A*0114", "A*2610", "A*2619", "A*3202", "A*3604", "A*2502", "A*7410", "A*2602", "A*0106", "A*0115", "A*2611", 
                                        "A*2621", "A*3205", "A*2504", "A*8001", "A*2603", "A*0107", "A*2604", "A*2612", "A*2623", "A*3206", "A*2622", 
                                        "A*3002", "A*0108", "A*2605", "A*2613", "A*2624", "A*3207", "A*3110", "A*3003", "A*0109", "A*2606", "A*2614", 
                                        "A*2626", "A*3209", "A*3203", "A*3004", "A*0110", "A*2607", "A*2615", "A*3006", "A*3210", "A*3204", "A*3201", 
                                        "A*0111", "A*2608", "A*2617", "A*3009", "A*3601", "A*3208"),'A01',
                    ifelse(HLA_type_new %in% c("A*3001", "A*3008", "A*3011", "A*3014", "A*3015", "A*0252", "A*3013", "A*6806", "A*6807"),'A01 A03',
                           ifelse(HLA_type_new %in% c("A*2902", "A*2901", "A*2905", "A*2909", "A*2911", "A*2913", "A*2903", "A*2906", "A*2910", "A*2912"),'A01 A24',
                                  ifelse(HLA_type_new %in% c("A*0201", "A*0209", "A*0224", "A*0240", "A*0257", "A*0271", "A*6827", "A*0241", "A*0202", "A*0211", 
                                                             "A*0225", "A*0243", "A*0258", "A*0272","A*6828", "A*0242", "A*0203", "A*0212", "A*0226", "A*0244", 
                                                             "A*0259","A*0274", "A*0250", "A*0204", "A*0213", "A*0227", "A*0245", "A*0261", "A*0275", "A*0260", 
                                                             "A*0205", "A*0215", "A*0228", "A*0246", "A*0262","A*0277", "A*0273", "A*0206", "A*0216", "A*0230", 
                                                             "A*0247", "A*0263", "A*0278", "A*0284", "A*0207", "A*0218", "A*0231", "A*0248", "A*0266", "A*0279",
                                                             "A*6815", "A*0214", "A*0219", "A*0236", "A*0249", "A*0267",  "A*0282", "A*0217", "A*0220", "A*0237",
                                                             "A*0251", "A*0268", "A*0283",  "A*6802", "A*0221", "A*0238", "A*0254", "A*0269", "A*0285", "A*6901",
                                                             "A*0222", "A*0239", "A*0256", "A*0270", "A*0286"),'A02',
                                         ifelse(HLA_type_new %in% c("A*0301", "A*0302", "A*0316", "A*1112", "A*3105", "A*3404", "A*6812", "A*7402", "A*0265", "A*1101", 
                                                                    "A*0304", "A*0317", "A*1113", "A*3106", "A*3406", "A*6813", "A*7403", "A*0280", "A*3101", "A*0305",
                                                                    "A*1102", "A*1114", "A*3109", "A*6602", "A*6814", "A*7404", "A*0309", "A*3301", "A*0306", "A*1103",
                                                                    "A*1115", "A*3111", "A*6603", "A*6816", "A*7405", "A*1106", "A*3303", "A*0307", "A*1104", "A*1116",
                                                                    "A*3304", "A*6604", "A*6819", "A*7407", "A*1122", "A*6601", "A*0308", "A*1105", "A*1120", "A*3305",
                                                                    "A*6803", "A*6821", "A*7408", "A*3112", "A*6801", "A*0310", "A*1107", "A*1121", "A*3306", "A*6804",
                                                                    "A*6822", "A*7409", "A*6805", "A*7401", "A*0312", "A*1108", "A*1123", "A*3307", "A*6808", "A*6824",
                                                                    "A*7411", "A*6820", "A*0313", "A*1109", "A*3103", "A*3402", "A*6809", "A*6825", "A*6823", "A*0314",
                                                                    "A*1110", "A*3104", "A*3403", "A*6810", "A*6826", "A*7406"), 'A03',
                                                ifelse(HLA_type_new %in% c("A*2301", "A*2302", "A*2310", "A*2410", "A*2422", "A*2433", "A*2440", "A*2305", "A*2442", 
                                                                           "A*2402", "A*2303", "A*2403", "A*2411", "A*2423", "A*2434", "A*2443", "A*2312", "A*2444", 
                                                                           "A*2304", "A*2405", "A*2413", "A*2426", "A*2435", "A*2446", "A*2417", "A*2452", "A*2306", 
                                                                           "A*2406", "A*2418", "A*2427", "A*2437", "A*2447", "A*2425", "A*2307", "A*2408", "A*2420", 
                                                                           "A*2428", "A*2438", "A*2448", "A*2430", "A*2308", "A*2409", "A*2421", "A*2429", "A*2439", "A*2449", "A*2441"),'A24',
                                                       ifelse(HLA_type_new %in% c("A*0102", "A*0233", "A*0276", "A*1118", "A*2414", "A*2432", "A*2616", "A*3007","A*3308",
                                                                                  "A*0113", "A*0234", "A*0281", "A*1119", "A*2415", "A*2450", "A*2620","A*3010", "A*3401",
                                                                                  "A*0208", "A*0235", "A*0315", "A*2309", "A*2419", "A*2451", "A*2904", "A*3102", "A*3405", 
                                                                                  "A*0210", "A*0255", "A*1111", "A*2404", "A*2424", "A*2453", "A*2907", "A*3107", "A*4301", 
                                                                                  "A*0229", "A*0264", "A*1117", "A*2407", "A*2431", "A*2503", "A*2914", "A*3108", "A*6817"),'A-none',
                                                              
                                                              ifelse(HLA_type_new %in% c("B*0702", "B*0704", "B*0741", "B*3540", "B*5109", "B*5134", "B*5510", "B*0707", "B*3518", "B*5112", "B*8102", 
                                                                                         "B*0703", "B*0706", "B*0742", "B*3541", "B*5110", "B*5135", "B*5515", "B*0709", "B*3529", "B*5113", "B*0705", 
                                                                                         "B*0715", "B*0743", "B*3542", "B*5111", "B*5136", "B*5517", "B*0712", "B*3530", "B*5114", "B*1508", "B*0719", 
                                                                                         "B*3507", "B*3543", "B*5116", "B*5138", "B*5519", "B*0714", "B*3534", "B*5120", "B*3501", "B*0720", "B*3508", 
                                                                                         "B*3544", "B*5117", "B*5302", "B*5603", "B*0716", "B*3537", "B*5137", "B*3503", "B*0721", "B*3511", "B*3545",
                                                                                         "B*5118", "B*5306", "B*5605", "B*0717", "B*3539", "B*5304", "B*4201", "B*0722", "B*3514", "B*3546", "B*5119", 
                                                                                         "B*5308", "B*5613", "B*0718", "B*3551", "B*5508", "B*5101", "B*0724", "B*3515", "B*3554", "B*5121", "B*5310",
                                                                                         "B*5615", "B*0723", "B*3553", "B*5511", "B*5102", "B*0725", "B*3521", "B*3555", "B*5123", "B*5403", "B*5616",
                                                                                         "B*0736", "B*3558", "B*5513", "B*5103", "B*0726", "B*3522", "B*3557", "B*5124", "B*5404", "B*7802", "B*0737", 
                                                                                         "B*3560", "B*5514", "B*5301", "B*0730", "B*3524", "B*3561", "B*5126", "B*5406", "B*7804", "B*3502", "B*3806", 
                                                                                         "B*5602", "B*5401", "B*0731", "B*3531", "B*3910", "B*5128", "B*5407", "B*3504", "B*3807", "B*5604", "B*5501", 
                                                                                         "B*0733", "B*3532", "B*3916", "B*5129", "B*5503", "B*3505", "B*3917", "B*5609", "B*5502", "B*0734", "B*3533", 
                                                                                         "B*4204", "B*5130", "B*5504", "B*3506", "B*4206", "B*5610", "B*5601", "B*0735", "B*3535", "B*4205", "B*5131", 
                                                                                         "B*5505", "B*3509", "B*4406", "B*5611", "B*6701", "B*0739", "B*3536", "B*5105", "B*5132", "B*5507", "B*3512", 
                                                                                         "B*5104", "B*5612", "B*7801", "B*0740", "B*3538", "B*5108", "B*5133", "B*5509", "B*3517", "B*5106", "B*8101" ),'B07' ,
                                                                     ifelse(HLA_type_new %in% c("B*0801", "B*0807", "B*0811", "B*0815", "B*0819", "B*0821", "B*0823", "B*0825", "B*0803", "B*0812", "B*0802", 
                                                                                                "B*0809", "B*0813", "B*0818", "B*0820", "B*0822", "B*0824", "B*0808", "B*0816"),'B08',
                                                                            ifelse(HLA_type_new %in% c("B*1402", "B*1401", "B*1593", "B*3926", "B*1405", "B*3913", "B*5518", "B*1503", "B*1403", "B*1598", "B*3927", 
                                                                                                       "B*1523", "B*3915", "B*1509", "B*1406", "B*1599", "B*3929", "B*1568", "B*3924", "B*1510", "B*1407", "B*2710", 
                                                                                                       "B*3930", "B*2701", "B*3928", "B*1518", "B*1537", "B*2713", "B*3932", "B*2711", "B*3933", "B*2702", "B*1547", 
                                                                                                       "B*2715", "B*4012", "B*2714", "B*3934", "B*2703", "B*1549", "B*2717", "B*4802", "B*2719", "B*4440", "B*2704", 
                                                                                                       "B*1551", "B*2725", "B*4803", "B*2720", "B*4807", "B*2705", "B*1552", "B*2728", "B*4804", "B*2721", "B*4808",
                                                                                                       "B*2706", "B*1554", "B*3805", "B*4805", "B*2724", "B*2707", "B*1561", "B*3809", "B*4809", "B*2727", "B*2709", 
                                                                                                       "B*1562", "B*3810", "B*4810", "B*2730", "B*3801", "B*1569", "B*3811", "B*4811", "B*3526", "B*3901", "B*1572", 
                                                                                                       "B*3904", "B*4812", "B*3903", "B*3902", "B*1574", "B*3907", "B*4813", "B*3905", "B*3909", "B*1580", "B*3914", 
                                                                                                       "B*9503", "B*3906", "B*4801", "B*1590", "B*3918", "B*3908", "B*7301", "B*1591", "B*3923", "B*3911"),'B27',
                                                                                   ifelse(HLA_type_new %in% c("B*1801", "B*1553", "B*1820", "B*4029", "B*4056", "B*4422", "B*4435", "B*5001", "B*1546", 
                                                                                                              "B*4028", "B*4048", "B*4420", "B*3701", "B*1803", "B*3704", "B*4035", "B*4057", "B*4424", 
                                                                                                              "B*4436", "B*5002", "B*1802", "B*4030", "B*4051", "B*4425", "B*4001", "B*1805", "B*4005", 
                                                                                                              "B*4039", "B*4102", "B*4426", "B*4437", "B*5004", "B*1814", "B*4033", "B*4052", "B*4431", 
                                                                                                              "B*4002", "B*1806", "B*4011", "B*4040", "B*4103", "B*4427", "B*4438", "B*4003", "B*4034", 
                                                                                                              "B*4058", "B*4434", "B*4006", "B*1810", "B*4014", "B*4049", "B*4404", "B*4428", "B*4503", 
                                                                                                              "B*4004", "B*4036", "B*4059", "B*4439", "B*4402", "B*1811", "B*4015", "B*4050", "B*4407",
                                                                                                              "B*4429", "B*4504", "B*4009", "B*4038", "B*4101", "B*4441", "B*4403", "B*1813", "B*4016",
                                                                                                              "B*4053", "B*4413", "B*4430", "B*4505", "B*4010", "B*4042", "B*4104", "B*4442", "B*4501", 
                                                                                                              "B*1815", "B*4020", "B*4054", "B*4416", "B*4432", "B*4507", "B*4018", "B*4043", "B*4106",
                                                                                                              "B*4502", "B*1819", "B*4026", "B*4055", "B*4421", "B*4433", "B*4904", "B*4019", "B*4044", 
                                                                                                              "B*4107", "B*4704", "B*4023", "B*4045", "B*4405", "B*4705", "B*4024", "B*4047", "B*4414"),'B44',
                                                                                          ifelse(HLA_type_new %in% c("B*1516", "B*1567", "B*5804", "B*5704", "B*1517", "B*1595", "B*5806", "B*5705", 
                                                                                                                     "B*5701", "B*5703", "B*5807", "B*5706", "B*5702", "B*5707", "B*5808", "B*5805", 
                                                                                                                     "B*5801", "B*5708", "B*5809", "B*5802", "B*5709", "B*5811"),'B58',
                                                                                                 ifelse(HLA_type_new %in% c("B*1501", "B*1505", "B*1530", "B*1548", "B*1578", "B*1596", "B*5202", 
                                                                                                                            "B*9502", "B*1504", "B*1558", "B*1309", "B*1502", "B*1514", "B*1531", 
                                                                                                                            "B*1550", "B*1581", "B*1597", "B*5203", "B*9504", "B*1507", "B*1573", 
                                                                                                                            "B*1313", "B*1512", "B*1515", "B*1533", "B*1560", "B*1582", "B*3528", 
                                                                                                                            "B*5204", "B*1524", "B*1586", "B*1513", "B*1519", "B*1534", "B*1563", 
                                                                                                                            "B*1583", "B*4021", "B*5205", "B*1535", "B*4408", "B*4601", "B*1520", 
                                                                                                                            "B*1538", "B*1565", "B*1585", "B*4603", "B*5207", "B*1542", "B*4602", 
                                                                                                                            "B*5201", "B*1525", "B*1539", "B*1570", "B*1588", "B*4604", "B*5208", 
                                                                                                                            "B*1545", "B*1528", "B*1540", "B*1575", "B*1592", "B*4605", "B*7805", "B*1555"),'B62','B-none'
                                                                                                 )
                                                                                          )
                                                                                   )
                                                                            )
                                                                            
                                                                     )      
                                                              )
                                                       )
                                                       
                                                       
                                                       
                                                       
                                                       
                                                       
                                                )
                                         ) 
                                  )
                           )
                    )
  )
  
) 


result_super <-result_total %>% dplyr::select(Supertype,HLA_type, Frequency,HLA_type_new) %>% arrange(Supertype, HLA_type, desc(Frequency))


result_super = result_super[!duplicated(result_super[,c('HLA_type')]),]

result_super<-result_super %>% dplyr::mutate(
  Degree=ifelse(HLA_type_new %in% c("A*0101", "A*2601", "A*2602", "A*2603", "A*3002", "A*3003", "A*3004", "A*3201", 
                                    "A*3001", "A2902", "A*0201", "A*0202", "A*0203", "A*0204",  "A*0205", "A*0206", "A*0207", 
                                    "A*0214", "A*0217", "A*6202","A*6901", "A*0301","A*1101","A*3101","A*3301","A*3303","A*6601","A*6801","A*7401","A*2301","A*2402",
                                    "B*0702","B*0703","B*0705","B*1508","B*3501","B*3503","B*4201","B*5101","B*5102","B*5103","B*5301","B*5401","B*5501",
                                    "B*5502","B*5601","B*6701","B*7801","B*0801","B*0802", "B*1402", "B*1503", "B*1509", "B*1510", "B*1518", "B*2702",
                                    "B*2703", "B*2704", "B*2705", "B*2706", "B*2707", "B*2709", "B*3801", "B*3901", "B*3902", "B*3909", "B*4801", "B*7301",
                                    "B*1801", "B*3701", "B*4001", "B*4002", "B*4006", "B*4402", "B*4403", "B*4501", "B*1516", "B*1517", "B*5701", "B*5702", "B*5801", "B*5802",
                                    "B*1501", "B*1502", "B*1512", "B*1513", "B*4601", "B*5201"),'High' ,
                ifelse(HLA_type_new %in% c("A*2501", "A*3603", "A*2502", "A*7410", "A*2504", "A*8001", "A*2622", "A*3110", "A*3203", "A*3204", "A*3208", "A*0252",
                                           "A*3013", "A*6806", "A*6807", "A*2913", "A*0241", "A*0242", "A*0250", "A*0260", "A*0273", "A*0284", "A*6815", "A*0265", 
                                           "A*0280", "A*0309", "A*1106", "A*1122", "A*3112", "A*6805", "A*6820", "A*6823", "A*7406", "A*2305", "A*2442", "A*2312", 
                                           "A*2444", "A*2417", "A*2452", "A*2425", "A*2430", "A*2441", "B*0707", "B*3518", "B*5112", "B*8102", "B*0709", "B*3529", 
                                           "B*5113", "B*0712", "B*3530", "B*5114", "B*0714", "B*3534", "B*5120", "B*0716", "B*3537", "B*5137", "B*0717", "B*3539", 
                                           "B*5304", "B*0718", "B*3551", "B*5508", "B*0723", "B*3553", "B*5511", "B*0736", "B*3558", "B*5513", "B*0737", "B*3560", 
                                           "B*5514", "B*3502", "B*3806", "B*5602", "B*3504", "B*3807", "B*5604 ", "B*3505", "B*3917", "B*5609", "B*3506", "B*4206",
                                           "B*5610", "B*3509", "B*4406", "B*5611", "B*3512", "B*5104", "B*5612", "B*3517", "B*5106", "B*8101", "B*0803", "B*0812", 
                                           "B*0808", "B*0816", "B*1405", "B*3913", "B*1523", "B*3915", "B*1568", "B*3924", "B*2701", "B*3928", "B*2711", "B*3933", 
                                           "B*2714", "B*3934", "B*2719", "B*4440", "B*2720", "B*4807", "B*2721", "B*4808", "B*2724 ", "B*2727 ", "B*2730 ", "B*3526 ",
                                           "B*3903 ", "B*3905 ", "B*3906 ", "B*3908 ", "B*3911", "B*1546", "B*4028", "B*4048", "B*4420", "B*1802", "B*4030", "B*4051", 
                                           "B*4425", "B*1814", "B*4033", "B*4052", "B*4431", "B*4003", "B*4034", "B*4058", "B*4434", "B*4004", "B*4036", "B*4059", "B*4439",
                                           "B*4009", "B*4038", "B*4101", "B*4441", "B*4010", "B*4042", "B*4104", "B*4442", "B*4018", "B*4043", "B*4106", "B*4502", "B*4019",
                                           "B*4044", "B*4107", "B*4704", "B*4023", "B*4045", "B*4405", "B*4705", "B*4024", "B*4047", "B*4414", "B*5704", "B*5705", "B*5706", 
                                           "B*5805", "B*1504", "B*1558", "B*1507", "B*1573", "B*1524", "B*1586", "B*1535", "B*4408", "B*1542", "B*4602", "B*1545", "B*1555"),'Low',
                       ifelse(HLA_type_new %in% c("A*0102", "A*0233", "A*0276", "A*1118", "A*2414", "A*2432", "A*2616", "A*3007", "A*3308", "A*0113", "A*0234", "A*0281", 
                                                  "A*1119", "A*2415", "A*2450", "A*2620", "A*3010", "A*3401", "A*0208", "A*0235", "A*0315", "A*2309", "A*2419", "A*2451",
                                                  "A*2904", "A*3102", "A*3405", "A*0210", "A*0255", "A*1111", "A*2404", "A*2424", "A*2453", "A*2907", "A*3107", "A*4301", 
                                                  "A*0229", "A*0264", "A*1117", "A*2407", "A*2431", "A*2503", "A*2914", "A*3108", "A*6817", "B*0708", "B*0806", "B*1310", 
                                                  "B*1543", "B*1589", "B*2723", "B*3547", "B*3709", "B*4008", "B*4410", "B*4901", "B*5402", "B*8201", "B*0710", "B*0810", 
                                                  "B*1311", "B*1544", "B*1804", "B*2726", "B*3548", "B*3802", "B*4013", "B*4411", "B*4902", "B*5405", "B*8202", "B*0711", 
                                                  "B*0814", "B*1312", "B*1556", "B*1807", "B*2729", "B*3549", "B*3803", "B*4025", "B*4412", "B*4903", "B*5512", "B*8301",
                                                  "B*0713", "B*0817", "B*1404", "B*1557", "B*1808", "B*3510", "B*3550", "B*3804", "B*4027", "B*4415", "B*5107", "B*5516", 
                                                  "B*9501", "B*0727", "B*1301", "B*1506", "B*1564", "B*1809", "B*3513", "B*3552", "B*3808", "B*4037", "B*4417", "B*5115", 
                                                  "B*5606", "B*0728", "B*1302", "B*1511", "B*1566", "B*1812", "B*3516", "B*3556", "B*3912", "B*4046", "B*4418", "B*5122", 
                                                  "B*5607", "B*0729", "B*1303", "B*1521", "B*1571", "B*1818", "B*3519", "B*3702", "B*3919", "B*4060", "B*4506", "B*5206", 
                                                  "B*5608", "B*0732", "B*1304", "B*1527", "B*1576", "B*2708", "B*3520", "B*3705", "B*3920", "B*4061", "B*4701", "B*5303", 
                                                  "B*5614", "B*0738", "B*1306", "B*1529", "B*1577", "B*2712", "B*3523", "B*3706", "B*3922", "B*4105", "B*4702", "B*5305", 
                                                  "B*5901", "B*0804", "B*1307", "B*1532", "B*1584", "B*2716", "B*3525", "B*3707", "B*3931", "B*4202", "B*4703", "B*5307", 
                                                  "B*6702", "B*0805", "B*1308", "B*1536", "B*1587", "B*2718", "B*3527", "B*3708", "B*4007", "B*4409", "B*4806", "B*5309", "B*7803"),NA ,'Middle')
                )
  )
) %>% dplyr::select(-HLA_type_new)


result_total<-result_total %>% dplyr::select(-HLA_type_new)

result_super$Supertype<- ifelse((result_super$Supertype=='B-none')&(substr(result_super$HLA_type,1,1)=='A'),'A-other',result_super$Supertype)
result_super$Supertype<- ifelse((result_super$Supertype=='B-none')&(result_super$Degree=='Middle'),'B-other',result_super$Supertype)
result_super$Supertype<- ifelse(is.na(result_super$Supertype),'B-None',result_super$Supertype)
result_super$Supertype<- ifelse((substr(result_super$HLA_type,1,1)=='C'),'C-other',result_super$Supertype)

result_super$Degree<- ifelse(result_super$Supertype=='A-other',NA, result_super$Degree)
result_super$Degree<- ifelse(result_super$Supertype=='B-other',NA, result_super$Degree)
result_super$Degree<- ifelse(result_super$Supertype=='C-other',NA, result_super$Degree)



result_super<-result_super %>% arrange(Supertype,HLA_type)

write.table(result_total, "C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/output6.txt",
            sep = "\t",
            row.names = F)

write.table(result_super, "C:/Users/afeve/Documents/Tutorials/Geneset/HLA dataset/output7.txt",
            sep = "\t",
            row.names = F)

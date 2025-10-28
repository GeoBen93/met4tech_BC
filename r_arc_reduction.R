library(maps) #maps library loads first so purr:map isn't overwritten
library(tidyverse)
library(glue)
library(colorspace)
library(boorm)
library(scatterpie) # from https://github.com/GuangchuangYu/scatterpie



################################################################################
# Helper Functions
################################################################################

# remove wt% and ppm from string names
fixName <- function (x) {
  
  #Full list of all elements in GEOROC
  dict <- c("SiO2","TiO2","Al2O3","Fe2O3","FeO","MnO","MgO","CaO","Na2O","K2O","FeOt",  
            "Fe2O3t","Li2O","P2O5","Cr2O3","NiO","LOI","H2O","H2O_MINUS","H2O_PLUS",
            "CO2","F","B","S","mg#","Ac","Ag","Al","As","At","Au","Ba","Be","Bi","Br",
            "Ca","Cd","Ce","Cl","Co","Cr","Cs","Cu","Dy","Er","Eu","Fe","Ga","Gd","Ge",
            "Hf","Hg","Ho","In","Ir","La","Li","Lu","Mg","Mn","Mo","Na","Nb","Nd","Ne",
            "Ni","Np","Os","Pa","Pb","Pd","Pm","Pr","Pt","Pu","Rb","Re","Rh","Ru","Sb",
            "Sc","Se","Si","Sm","Sn","Sr","Ta","Tb","Te","Th","Ti","Tl","Tm","U","V",
            "W","Y","Yb","Zn","Zr")
  
  clean_name <- x %>%
    str_remove("\\(WT%\\)") %>% 
    str_remove("\\(PPM\\)") 
  
  p = match( clean_name, toupper(dict) )
  if ( !is.na(p) ) {
    return( dict[p] )
  }
  return(clean_name)
}


CleanData <- function(data, elements_to_clean){
  x <- data %>%
    #remove wt% and ppm
    dplyr::rename_with(
      function(x){ mapply(fixName, x) }
    ) %>%
    
    dplyr::select(all_of(elements_to_clean)) %>%
    tidyr::drop_na()
  
  x %>%
    filter(LOCATION %in% (
      x %>%
        group_by(LOCATION) %>%
        dplyr::tally() %>%
        dplyr::filter(n >= 10) %>%
        dplyr::select(LOCATION) %>%
        unlist()  
    ))
}

AddNorms <- function(data, elements_to_norm){
  data %>% dplyr::mutate(across(elements_to_norm, ~ ifelse(.x > 0, log(.x / (exp(mean(log(.x[.x > 0]))))), 0), .names = "norm_{.col}" ))
}


# name the quartiles
QuartileCalc <- function(x, y) {
  if (is.na(x + y)){
    return(NA)
  }
  if(x >= 0){
    if (y >= 0){
      return("+ve/+ve")
    }
    return("+ve/-ve")
  }
  if(y >= 0) {
    return("-ve/+ve")
  }
  return("-ve/-ve")
}

# from https://davetang.org/muse/2012/02/10/manual-linear-regression-analysis-using-r/

# helper functions to make the walrus operator expressions more readable
slope <- function( x, y ) {
  mean_x <- mean(x)
  return(sum((x-mean_x)*(y-mean(y)))/sum((x-mean_x)^2))
}

# this is not required but shows what can be done

intercept <- function( x, y, c = 0 ) {
  m <- slope(x,y)
  return (m*c + mean(y)-m*mean(x))
}

sample_summary <- function(data, x, y) {
  data %>%
    group_by(LOCATION) %>% 
    summarise("mean_{{x}}" := mean({{x}}),
              "mean_{{y}}" := mean({{y}}),
              "R2_{{x}}vs{{y}}" := cor({{x}}, {{y}})^2,
              "slope_{{x}}vs{{y}}" := slope({{x}},{{y}}),
              "intercept0_{{x}}vs{{y}}" := intercept({{x}},{{y}},0),
              "intercept45_{{x}}vs{{y}}" := intercept({{x}},{{y}},45)
    )
}

# x y quandrant colour mapping 

assign_colour_XY_map <- function(colour_map) {
  
  function(x, y) {
    
    theta <- ifelse(x < 0, atan(y/x)+pi, ifelse(y < 0, atan(y/x)+2*pi, atan(y/x)))
    magnitude <- sqrt(x^2 + y^2)
    seg <- theta%/%(pi/2)
    arc <- theta%%(pi/2)
    
    if (arc < (pi/8)) {
      prev_seg = (seg+3)%%4
      return(hex(colorspace::mixcolor(0.5 + 0.5*arc/(pi/8), colour_map[[prev_seg+1]], colour_map[[seg+1]]) ))
    } else if (arc < (pi/2-pi/8)) {
      return(hex(colour_map[[seg+1]]))
    } else {
      next_seg = (seg+1)%%4
      return(hex(colorspace::mixcolor(0.5*((pi/8)-(pi/2-arc))/(pi/8), colour_map[[seg+1]], colour_map[[next_seg+1]]) ))
    }
  }
}

#colour map with radius white scale
assign_colour_XY_map <- function(colour_map) {
  
  function(x, y) {
    
    theta <- ifelse(x < 0, atan(y/x)+pi, ifelse(y < 0, atan(y/x)+2*pi, atan(y/x)))
    magnitude <- sqrt(x^2 + y^2)
    seg <- theta%/%(pi/2)
    arc <- theta%%(pi/2)
    
    if (arc < (pi/8)) {
      prev_seg = (seg+3)%%4
      colour <- colorspace::mixcolor(0.5 + 0.5*arc/(pi/8), colour_map[[prev_seg+1]], colour_map[[seg+1]]) 
    } else if (arc < (pi/2-pi/8)) {
      colour <- colour_map[[seg+1]]
    } else {
      next_seg = (seg+1)%%4
      colour <- colorspace::mixcolor(0.5*((pi/8)-(pi/2-arc))/(pi/8), colour_map[[seg+1]], colour_map[[next_seg+1]]) 
    }
    if (magnitude*2 > 1) {
      return(hex(colour))
    } else {
      return(hex(colorspace::mixcolor(magnitude*2, sRGB(1,1,1), colour)) )
    }
  }
}

# create custom mapping for colours (Below order is: Red, yellow, green, Blue)
assign_colour <- assign_colour_XY_map( c( sRGB(1,0,0), sRGB(1,0.8274,0), sRGB(0,1,0), sRGB(0.2823,0.05882,1)) )


#slope vs SiO2
slope_quartile <- function(data, x, y) {
  data %>%
    group_by(LOCATION) %>% 
    summarise(x = slope(SiO2,{{x}}),
              y = slope(SiO2,{{y}})
    ) %>%
    dplyr::mutate(cat = mapply(QuartileCalc, x, y)) # add cat column but not required
}


################################################################################
# Processing Starts Here
################################################################################

# Full list of elements to be separated
ELEMENTS <- c("SiO2", "TiO2", "Al2O3","Fe2O3", "FeO",  "FeOt", "MnO" ,"MgO",
              "CaO", "Na2O", "K2O", "P2O5", "Cs", "Rb", "Ba", "Th", "U", "Nb", "Ta", "La",
              "Ce", "Pb", "Pr", "Sr", "Nd", "Zr", "Hf", "Sm", "Eu", "Gd", "Tb", "Dy", "Y",
              "Ho", "Er", "Yb", "Lu")


# Short list for tests. I used the shorter one for most tests to save performence.
ELEMENTS <- c("SiO2","Sr", "Y", "MnO")

#small_data <- read_csv("nick/MARIANA_ARC.csv")

#This should be changed to match the path of your own data
full_data <- read_csv("nick/R_arcs - DO NOT EDIT.csv")


##################
#Create baseline data sets
#Process can vary depending on input data source, so this section may need modification depending on format.
#Ideally sorted_data should be a location column, followed by a column for each element. 
#Avoid element names having symbols such as "%" or spaces

sorted_data <- full_data %>%
  CleanData(c("LOCATION", ELEMENTS)) %>%
  AddNorms(ELEMENTS)  


################
##Plot Examples#
################



#simple plot of the slope Sr vs slope of Y using 4 colour mapping
#x= slope_Sr, y= slope_Y
sorted_data %>%
  slope_quartile( Sr, Y ) %>%
  ggplot(aes(x = x, y = y, colour = cat)) + 
  geom_point() +
  scale_colour_identity() +
  xlim(-150,150) +
  ylim(-10,10)

#plot using the full colour graident mapping 
sorted_data %>%
  slope_quartile( Sr, Y ) %>%
  ggplot(aes(x = x, y = y, colour = mapply(assign_colour, x/30, y/2))) + # note 30:2 scaling for colour shading to match 150:10 crop limits
  geom_point() +
  scale_colour_identity() + 
  xlim(-150,150) +
  ylim(-10,10)


#SiO2 vs Sr/Y plot
sorted_data %>%
  dplyr::filter(str_detect(LOCATION, "ANDEAN ARC")) %>% # the text search is case sensitive
  dplyr::inner_join(slope_quartile(sorted_data, Sr, Y ), by = "LOCATION") %>%  
  #dplyr::filter(cat == "-ve/+ve") %>% # can be used to separate out individual quartiles if needed
  ggplot(aes(x = SiO2, y = Sr/Y, colour = mapply(assign_colour, x/30, y/2))) + # note 30:2 scaling for colour shading to match 150:10 crop limits
  geom_point(alpha = 0.8) +
  scale_colour_identity() +
  xlim(35, 80) +
  ylim(0, 100) 

#Bar chart showing absolute count of sample quartiles by major location
#Colour mapping does not work for this, correct colour for each segment is in the key
sorted_data %>%
  dplyr::inner_join(slope_quartile(sorted_data, Sr, Y ), by = "LOCATION") %>%
  dplyr::rename(slope_SiO2vsSr = x, slope_SiO2vsY = y) %>%
  dplyr::mutate(major_loc = sub(" / .*", "", x = LOCATION)) %>%
  ggplot(aes(x=major_loc, fill = cat)) +
  scale_color_identity() +
  geom_bar() +
  theme(axis.text.x = element_text(angle = 45, hjust=1))

#Worldmap with piecharts for each major location
worldmap <- map_data("world")

major_location_coords <- full_data %>%
  dplyr::mutate(major_loc = sub(" / .*", "", x = LOCATION)) %>% #cuts the Location string before the first " / ".
  dplyr::group_by(major_loc) %>%
  summarise("major_latitude" = mean((`LATITUDE MIN`) + mean(`LATITUDE MAX`))/2,
            "major_longitude" = mean((`LONGITUDE MIN`) + mean(`LONGITUDE MAX`))/2)

major_location_stats <- sorted_data %>%
  dplyr::inner_join(slope_quartile(sorted_data, Sr, Y ), by = "LOCATION") %>%
  dplyr::rename(slope_SiO2vsSr = x, slope_SiO2vsY = y) %>%
  dplyr::mutate(major_loc = sub(" / .*", "", x = LOCATION)) %>%
  
  dplyr::group_by(major_loc) %>%
  count(cat) %>%
  pivot_wider(names_from = cat, values_from = n, values_fill = 0) %>%  
  dplyr::inner_join(major_location_coords, by = "major_loc") %>%
  dplyr::rename("+ve/+ve" = "#FF0000", "-ve/-ve" = "#00FF00", "-ve/+ve" = "#480FFF", "+ve/-ve" = "#FFD300")

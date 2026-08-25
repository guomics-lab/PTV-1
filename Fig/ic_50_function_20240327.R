fitModel <- function(gDat, vStart =c(8.886464, 1.495953 ) , bLargeScale=TRUE, bSilent=TRUE){#c(8.886464, 1.495953 )
  # start with sanity checks; catch most common mistakes early
  # first, do we have all requires columns?
  if(length(which(colnames(gDat)=='CL'))==0){
    stop('gDat is required to contain the cell lines names in a column with name CL')  
  }
  if(length(which(colnames(gDat)=='x'))==0){
    stop('gDat is required to contain the X-from concentration step names in a column with name x')  
  }
  if(length(which(colnames(gDat)=='y'))==0){
    stop('gDat is required to contain the relative kill (1-viability) in a column with name y')  
  }
  if(length(which(colnames(gDat)=='drug'))==0){
    stop('gDat is required to contain the drug in a column with name drug')  
  }
  if(length(which(colnames(gDat)=='maxc'))==0){
    stop('gDat is required to contain the maxc (maximum concentration) in a column with name maxc')  
  }
  # coding of relative kill (1-viability) correct; mean of gDat$y at gDat$x==9 should be higher than at min(gDat$x)
  tmp.minx <- min(gDat$x)
  tmp.whichxmin <- which(gDat$x == tmp.minx)
  tmp.whichxmax <- which(gDat$x == 7)
  tmp.ymin <- mean(stats::na.omit(gDat$y[tmp.whichxmin]))
  tmp.ymax <- mean(stats::na.omit(gDat$y[tmp.whichxmax]))
  # if(tmp.ymin > tmp.ymax){
  #   stop('Coding of relative viabilities seems incorrect; note that y is defined as 1-viability.')
  # }
  if(bLargeScale){#大规模数据处理模式 bSilent:以静默模式进行
    fmv5 <- nlme::nlme(y ~ logist3(ANCHOR_VIAB,x, xmid, scal),
                       fixed= xmid+scal~1,
                       random=list(CL = nlme::pdSymm(xmid+scal~1), 
                                   drug = nlme::pdDiag(xmid~1)),
                       data = gDat, start=vStart, method='REML')
    if(!bSilent){
      summary(fmv5)
    }
    return(fmv5)
  }else{
    fmv5 <- nlme::nlme(y ~ logist3(ANCHOR_VIAB,x, xmid, scal),
                       fixed = xmid + scal ~  1, 
                       random = list(CL = nlme::pdDiag(xmid + scal ~ 1),
                                     drug = nlme::pdDiag(xmid ~ 1)), 
                       data = gDat, start = vStart, method = "REML",
                       control = nlme::nlmeControl(pnlsTol = 0.2,
                                                   msVerbose = FALSE,
                                                   tolerance=1e-4,
                                                   returnObject = T)
    )
    if(!bSilent){
      summary(fmv5)
    }
    return(fmv5)
  }
}


logist3 <- stats::selfStart( ~ANCHOR_VIAB/(1 + exp(-(x - xmid)/scal)),#(1-ANCHOR_VIAB)+
                             initial = function(mCall, LHS, data){   
                               xy <- stats::sortedXyData(mCall[["x"]], LHS, data)
                               if(nrow(xy) < 3) {
                                 stop("Too few distinct input values to fit a logistic")
                               }
                               xmid <- stats::NLSstClosestX(xy, 0.5*ANCHOR_VIAB ) 
                               scal <- stats::NLSstClosestX(xy, 0.75*ANCHOR_VIAB ) - xmid
                               value <- c(xmid, scal)
                               names(value) <- mCall[c("xmid", "scal")]
                               value
                             },
                             parameters = c("xmid", "scal"))
# logist3(0.9,0.5,0.8,0.1)
# logist3(1,0.5,0.8,0.1)
# getConcFromX <- function(x, maxc) {
#   xc <- maxc * 2 ^ (x - 7)
#   return(xc)
# }



calcIC50 <- function(model_coef) {
  model_coef <- model_coef %>%
    mutate_(
      IC50 = ~log(getConcFromX(xmid, maxc)))
  return(model_coef)
}

calcAuc <- function(model_coef) {
  model_coef <- model_coef %>% 
    group_by_(~CL, ~drug) %>% 
    mutate_(xmin = ~min(x)) %>%
    mutate_(xmax = ~max(x)) %>%
    mutate_(auc = 
              ~1 - (getIntegral(xmax, xmid, scal) - 
                      getIntegral(xmin, xmid, scal)) / (xmax - xmin))
  model_coef <- model_coef %>% 
    ungroup() %>%
    select_(~-xmax, ~-xmin)
  return(model_coef)
}

calcAucTrap <- function(model_stats) {
  model_stats <- model_stats %>% 
    group_by_(~CL, ~drug) %>% arrange_(~x) %>% 
    mutate_(area = ~ (max(x) - min(x)) * max(c(1,yhat))) %>% 
    mutate_(AUCtrap = ~((caTools::trapz(x, 1 - yhat)) / area))
  model_stats <- model_stats %>% ungroup() %>% select_(~-area)
  return(model_stats)
}

calcNlmeFit <- function(model_coef){
  model_coef <- model_coef %>%
    group_by_(~CL, ~drug) %>%
    mutate_(yhat = ~logist3(ANCHOR_VIAB,x, xmid, scal))
  attributes(model_coef$yhat) <- NULL
  model_coef <- model_coef %>% mutate_(yres = ~y - yhat)
  model_coef <- model_coef %>% mutate_(RMSE = ~sqrt(mean(yres ^ 2)))
  model_coef <- model_coef %>% ungroup()
  return(model_coef)
}
calcNlmeStats <- function (nlme_model, nlme_data) {
  #   gDat <- groupNlmeData(nlme_data)
  model_coef <- getModelCoef(nlme_model = nlme_model, nlme_data = nlme_data)
  model_stats <- calcNlmeFit(model_coef)
  model_stats <- model_stats %>%
    arrange_(~desc(x)) %>%
    mutate_(x_micromol = ~getConcFromX(x, maxc))
  model_stats <- calcIC50(model_stats)
  model_stats <- calcAuc(model_stats)
  model_stats <- calcAucTrap(model_stats)
  return(model_stats)
}

getConcFromX <- function(x, maxc) {
  xc <- maxc * 2 ^ (x - 8)
  return(xc)
}

getXfromConc <- function(xc, maxc) {
  x <- (log(xc / maxc)/log(2))+ 8
  return(x)
}

l3_model2 <- function(ANCHOR_VIAB,lx, maxc, xmid, scal){
  x <- getXfromConc(exp(lx), maxc)
      # yhat <- ANCHOR_VIAB/(ANCHOR_VIAB + exp(ANCHOR_VIAB) ^ ((x - xmid) / scal))
  yhat <- ANCHOR_VIAB-logist3(ANCHOR_VIAB,x, xmid, scal)
  return(yhat)
}


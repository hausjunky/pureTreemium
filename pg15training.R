library(CASdatasets)
library(data.table)
library(xgboost)
library(foreach)
library(iterators)
library(parallel)
library(doParallel)

options(scipen = 999)
registerDoParallel(detectCores())

p <- c('data.table', 'xgboost')
f <- 5

data("pg15training")
df <- as.data.table(pg15training)
gc(rm(pg15training))

dups <- df[, .(count = .N), .(PolNum)]
dups <- dups[count > 1]$PolNum
dups <- df[PolNum %in% dups]
setorder(dups, PolNum)
dups <- dups[Indtppd > 0]
df <- df[!PolNum %in% dups$PolNum]
df <- rbind(df, dups)
setorder(df, PolNum)
gc(rm(dups))

df[, Gender := fifelse(Gender == "Male", 1, 0)]
df[, Type := as.integer(Type)]
df[, Category := fcase(Category == "Small", 1, Category == "Medium", 2, Category == "Large", 3, default = NA_integer_)]
df[, Occupation := as.integer(Occupation)]
df[, SubGroup2 := as.integer(SubGroup2)]
df[, Group2 := as.integer(Group2)]

pd <- df[CalYear == 2009]
bi <- df[CalYear == 2009]
pd[, Claims := Numtppd]
pd[, Amount := Indtppd]
pd[, c("Numtppd", "Numtpbi", "Indtppd", "Indtpbi") := NULL]
bi[, Claims := Numtpbi]
bi[, Amount := Indtpbi]
bi[, c("Numtppd", "Numtpbi", "Indtppd", "Indtpbi") := NULL]
setorder(pd, -Amount)
setorder(bi, -Amount)
pd[, Fold := (floor((1:nrow(pd)) / 5) %% 5) + 1]
bi[, Fold := (floor((1:nrow(bi)) / 5) %% 5) + 1]

vars <- setdiff(names(pd), c("PolNum", "CalYear", "Exppdays", "Claims", "Amount", "Fold"))

pd1 <- foreach(i = 1:5, .combine = rbind, .packages = p) %dopar% {
  
  vars <- vars
  
  trn <- pd[Fold != i]
  setorder(trn, -Claims)
  trn[, Fold := (floor((1:nrow(trn)) / 5) %% 5) + 1]
  fld <- list()
  for (j in 1:5) fld[[j]] <- which(trn$Fold == j)
  mtx <- xgb.DMatrix(as.matrix(trn[, ..vars]), label = fifelse(trn$Claims > 0, 1, 0))
  hyp <- list(objective = "binary:logistic", tree_method = "hist", grow_policy = "lossguide", eta = .01, max_leaves = 25, subsample = .25)
  xcv <- xgb.cv(hyp, mtx, 1e6, folds = fld, early_stopping_rounds = 10)
  bin <- xgb.train(hyp, mtx, xcv$best_iteration)
  
  trn <- pd[Fold != i & Amount > 1]
  setorder(trn, -Amount)
  trn[, Fold := (floor((1:nrow(trn)) / 5) %% 5) + 1]
  fld <- list()
  for (j in 1:5) fld[[j]] <- which(trn$Fold == j)
  mtx <- xgb.DMatrix(as.matrix(trn[, ..vars]), label = log(trn$Amount / trn$Exppdays))
  hyp <- list(objective = "reg:squarederror", tree_method = "hist", grow_policy = "lossguide", eta = .01, max_leaves = 25, subsample = .25)
  xcv <- xgb.cv(hyp, mtx, 1e6, folds = fld, early_stopping_rounds = 10)
  sev <- xgb.train(hyp, mtx, xcv$best_iteration)
  adj <- mean(trn$Amount / trn$Exppdays) / mean(exp(predict(sev, mtx)))
  
  tst <- pd[Fold == i]
  mtx <- xgb.DMatrix(as.matrix(tst[, ..vars]))
  tst$Binary <- predict(bin, mtx)
  tst$Severity <- exp(predict(sev, mtx)) * adj
  tst$Model <- tst$Binary * tst$Severity * tst$Exppdays
  tst <- tst[, .(PolNum, Binary, Severity, Model)]
  
}

bi1 <- foreach(i = 1:5, .combine = rbind, .packages = p) %dopar% {
  
  vars <- vars
  
  trn <- bi[Fold != i]
  setorder(trn, -Claims)
  trn[, Fold := (floor((1:nrow(trn)) / 5) %% 5) + 1]
  fld <- list()
  for (j in 1:5) fld[[j]] <- which(trn$Fold == j)
  mtx <- xgb.DMatrix(as.matrix(trn[, ..vars]), label = fifelse(trn$Claims > 0, 1, 0))
  hyp <- list(objective = "binary:logistic", tree_method = "hist", grow_policy = "lossguide", eta = .01, max_leaves = 25, subsample = .25)
  xcv <- xgb.cv(hyp, mtx, 1e6, folds = fld, early_stopping_rounds = 10)
  bin <- xgb.train(hyp, mtx, xcv$best_iteration)
  
  trn <- bi[Fold != i & Amount > 1]
  setorder(trn, -Amount)
  trn[, Fold := (floor((1:nrow(trn)) / 5) %% 5) + 1]
  fld <- list()
  for (j in 1:5) fld[[j]] <- which(trn$Fold == j)
  mtx <- xgb.DMatrix(as.matrix(trn[, ..vars]), label = log(trn$Amount / trn$Exppdays))
  hyp <- list(objective = "reg:squarederror", tree_method = "hist", grow_policy = "lossguide", eta = .01, max_leaves = 25, subsample = .25)
  xcv <- xgb.cv(hyp, mtx, 1e6, folds = fld, early_stopping_rounds = 10)
  sev <- xgb.train(hyp, mtx, xcv$best_iteration)
  adj <- mean(trn$Amount / trn$Exppdays) / mean(exp(predict(sev, mtx)))
  
  tst <- bi[Fold == i]
  mtx <- xgb.DMatrix(as.matrix(tst[, ..vars]))
  tst$Binary <- predict(bin, mtx)
  tst$Severity <- exp(predict(sev, mtx)) * adj
  tst$Model <- tst$Binary * tst$Severity * tst$Exppdays
  tst <- tst[, .(PolNum, Binary, Severity, Model)]
  
}

pd <- merge.data.table(pd, pd1, "PolNum")
bi <- merge.data.table(bi, bi1, "PolNum")

trn <- copy(pd)
setorder(trn, -Claims)
trn[, Fold := (floor((1:nrow(trn)) / 5) %% 5) + 1]
fld <- list()
for (j in 1:5) fld[[j]] <- which(trn$Fold == j)
mtx <- xgb.DMatrix(as.matrix(trn[, ..vars]), label = fifelse(trn$Claims > 0, 1, 0))
hyp <- list(objective = "binary:logistic", tree_method = "hist", grow_policy = "lossguide", eta = .01, max_leaves = 25, subsample = .25)
xcv <- xgb.cv(hyp, mtx, 1e6, folds = fld, early_stopping_rounds = 10)
pdb <- xgb.train(hyp, mtx, xcv$best_iteration)

trn <- pd[Amount > 1]
setorder(trn, -Amount)
trn[, Fold := (floor((1:nrow(trn)) / 5) %% 5) + 1]
fld <- list()
for (j in 1:5) fld[[j]] <- which(trn$Fold == j)
mtx <- xgb.DMatrix(as.matrix(trn[, ..vars]), label = log(trn$Amount / trn$Exppdays))
hyp <- list(objective = "reg:squarederror", tree_method = "hist", grow_policy = "lossguide", eta = .01, max_leaves = 25, subsample = .25)
xcv <- xgb.cv(hyp, mtx, 1e6, folds = fld, early_stopping_rounds = 10)
pds <- xgb.train(hyp, mtx, xcv$best_iteration)
pda <- mean(trn$Amount / trn$Exppdays) / mean(exp(predict(pds, mtx)))

trn <- copy(bi)
setorder(trn, -Claims)
trn[, Fold := (floor((1:nrow(trn)) / 5) %% 5) + 1]
fld <- list()
for (j in 1:5) fld[[j]] <- which(trn$Fold == j)
mtx <- xgb.DMatrix(as.matrix(trn[, ..vars]), label = fifelse(trn$Claims > 0, 1, 0))
hyp <- list(objective = "binary:logistic", tree_method = "hist", grow_policy = "lossguide", eta = .01, max_leaves = 25, subsample = .25)
xcv <- xgb.cv(hyp, mtx, 1e6, folds = fld, early_stopping_rounds = 10)
bib <- xgb.train(hyp, mtx, xcv$best_iteration)

trn <- bi[Amount > 1]
setorder(trn, -Amount)
trn[, Fold := (floor((1:nrow(trn)) / 5) %% 5) + 1]
fld <- list()
for (j in 1:5) fld[[j]] <- which(trn$Fold == j)
mtx <- xgb.DMatrix(as.matrix(trn[, ..vars]), label = log(trn$Amount / trn$Exppdays))
hyp <- list(objective = "reg:squarederror", tree_method = "hist", grow_policy = "lossguide", eta = .01, max_leaves = 25, subsample = .25)
xcv <- xgb.cv(hyp, mtx, 1e6, folds = fld, early_stopping_rounds = 10)
bis <- xgb.train(hyp, mtx, xcv$best_iteration)
bia <- mean(trn$Amount / trn$Exppdays) / mean(exp(predict(bis, mtx)))

fac <- 1 + (1 - mean(pd$Exppdays / 365))

df <- df[CalYear == 2010]
mtx <- xgb.DMatrix(as.matrix(df[, ..vars]))
df$BinaryPD <- predict(pdb, mtx)
df$SeverityPD <- exp(predict(pds, mtx)) * pda
df$ModelPD <- df$BinaryPD * df$SeverityPD * df$Exppdays
df$BinaryBI <- predict(bib, mtx)
df$SeverityBI <- exp(predict(bis, mtx)) * bia
df$ModelBI <- df$BinaryBI * df$SeverityBI * df$Exppdays
df$Model <- (df$ModelPD + df$ModelBI) * fac

sum(df$Indtppd + df$Indtpbi) / sum(df$Model)

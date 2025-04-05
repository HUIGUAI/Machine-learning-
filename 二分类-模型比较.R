# 模型机器---R语言tidymodels包机器学习分类与回归模型---二分类---模型比较

library(tidymodels)

# 加载各个模型的评估结果
evalfiles <- list.files(".\\cls2\\", full.names = T)
lapply(evalfiles, load, .GlobalEnv)

#############################################################

# 各个模型在测试集上的误差指标
eval <- bind_rows(
  eval_logistic, eval_enet, eval_dt,
  eval_rf, eval_xgboost, eval_rsvm, eval_mlp,
  eval_lightgbm, eval_knn
)
eval
# 平行线图
eval %>%
  filter(dataset == "test") %>%
  ggplot(aes(x = .metric, y = .estimate, color = model)) +
  geom_point() +
  geom_line(aes(group = model)) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
# 各个模型在测试集上的误差指标表格
eval2 <- eval %>%
  select(-.estimator) %>%
  filter(dataset == "test") %>%
  pivot_wider(names_from = .metric, values_from = .estimate)
eval2
# 各个模型在测试集上的误差指标图示
eval2 %>%
  ggplot(aes(x = model, y = roc_auc, fill = model)) +
  geom_col(width = 0.3, show.legend = F) +
  geom_text(aes(label = round(roc_auc, 2)), 
            nudge_y = -0.03) +
  theme_bw()

#############################################################

# 各个模型在测试集上的预测概率
predtest <- bind_rows(
  predtest_logistic, predtest_enet, predtest_dt,
  predtest_rf, predtest_xgboost, predtest_rsvm, predtest_mlp,
  predtest_lightgbm, predtest_knn
)
predtest


# 各个模型在测试集上的ROC
predtest %>%
  group_by(model) %>%
  roc_curve(AHD, .pred_No, event_level = "first") %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = model)) +
  geom_path(linewidth = 1) +
  theme_bw()

#############################################################

# 各个模型在测试集上的预测概率2
predtest2 <- predtest %>%
  select(-.pred_No) %>%
  mutate(id = rep(1:nrow(predtest_logistic), 9)) %>%
  pivot_wider(names_from = model, values_from = .pred_Yes)
predtest2


# 各个模型在测试集上的校准曲线
# http://topepo.github.io/caret/measuring-performance.html#calibration-curves
cal_obj <- caret::calibration(
  as.formula(paste0("AHD ~ ", 
                    paste(colnames(predtest2)[4:12], collapse = " + "))),
  data = predtest2,
  class = "Yes",
  cuts = 11
)
cal_obj$data
plot(cal_obj, type = "b", pch = 16,
     auto.key = list(columns = 3,
                     lines = T,
                     points = T))
cal_obj$data %>%
  filter(Percent != 0) %>%
  ggplot(aes(x = midpoint, y = Percent)) +
  geom_point(color = "brown1") +
  geom_line(color = "brown1") +
  geom_abline(slope = 1, intercept = 0) +
  facet_wrap(~calibModelVar) +
  theme_bw()

##############################

library(PredictABEL)
predtest4 <- predtest2
predtest4$AHD <- ifelse(predtest4$AHD == "No", 0, 1)
calall <- data.frame()
for (i in 4:12) {
  cal <- plotCalibration(data = as.data.frame(predtest4),
                         cOutcome = 1,
                         predRisk = predtest4[[i]])
  caldf <- cal$Table_HLtest %>%
    as.data.frame() %>%
    rownames_to_column("pi") %>%
    mutate(model = colnames(predtest4)[i])
  calall <- rbind(calall, caldf)
}
calall %>%
  ggplot(aes(x = meanpred, y = meanobs)) +
  geom_point(color = "brown1") +
  geom_line(color = "brown1") +
  geom_abline(slope = 1, intercept = 0) +
  facet_wrap(~model) +
  theme_bw()

#############################################################

# 各个模型在测试集上的DCA
# https://cran.r-project.org/web/packages/dcurves/vignettes/dca.html
dca_obj <- dcurves::dca(
  as.formula(paste0("AHD ~ ", 
                    paste(colnames(predtest2)[4:12], collapse = " + "))),
  data = predtest2,
  thresholds = seq(0, 1, by = 0.01)
)
plot(dca_obj, smooth = T)

#############################################################

# 各个模型交叉验证的各折指标点线图
evalcv <- bind_rows(
  eval_cv5_logistic, eval_best_cv5_enet, eval_best_cv5_dt,
  eval_best_cv5_rf, eval_best_cv5_xgboost, eval_best_cv5_rsvm,
  eval_best_cv5_mlp, eval_best_cv5_lightgbm, eval_best_cv5_knn
)
evalcv

evalcv %>%
  ggplot(aes(x = id, y = .estimate, 
             group = model, color = model)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(0.7, 1)) +
  labs(x = "", y = "roc_auc") +
  theme_bw()

# 各个模型交叉验证的指标平均值图(带上下限)
evalcv %>%
  group_by(model) %>%
  sample_n(size = 1) %>%
  ungroup() %>%
  ggplot(aes(x = model, y = mean)) +
  geom_point(size = 2) +
  # geom_line(group = 1) +
  geom_errorbar(aes(ymin = mean-std_err, 
                    ymax = mean+std_err),
                width = 0.1, linewidth = 1.2) +
  scale_y_continuous(limits = c(0.7, 1)) +
  labs(y = "cv roc_auc") +
  theme_bw()

#############################################################


# 校准曲线变种
predtest3 <- predtest %>%
  group_by(model) %>%
  arrange(model, .pred_Yes) %>%
  mutate(.pred_No = 1 - .pred_Yes,
         YesCount = cumsum(AHD == "Yes"),
         YesSum = sum(AHD == "Yes"),
         YesPct = YesCount / YesSum)
predtest3
predtest3 %>%
  ggplot(aes(x = .pred_Yes, y = YesPct, colour = model)) +
  geom_point() +
  geom_line(aes(group = model)) +
  theme_bw()


# ROC, NRI, IDI
library(jsmodule)
library(pROC)
m1 <- glm(vs ~ am + gear, data = mtcars, family = binomial)
m2 <- glm(vs ~ am + gear + wt, data = mtcars, family = binomial)
m3 <- glm(vs ~ am + gear + wt + mpg, data = mtcars, family = binomial)
roc1 <- roc(m1$y, predict(m1, type = "response"))
plot(roc1)
roc2 <- roc(m2$y, predict(m2, type = "response"))
plot(roc2, add = T, col = "blue")
roc3 <- roc(m3$y, predict(m3, type = "response"))
plot(roc3, add = T, col = "red")

list.roc <- list(roc1, roc2, roc3)
ROC_table(list.roc)



##############################################################





# 模型机器---R语言tidymodels包机器学习分类与回归模型---二分类---logistic回归

# https://www.tidymodels.org/find/parsnip/
# https://parsnip.tidymodels.org/reference/logistic_reg.html
# https://parsnip.tidymodels.org/reference/details_logistic_reg_glm.html

# 模型评估指标
# https://cran.r-project.org/web/packages/yardstick/vignettes/metric-types.html


library(tidymodels)

# 读取数据
Heart <- readr::read_csv(file.choose()) # tibble
colnames(Heart)
# 修正变量类型
# 将分类变量转换为factor
for(i in c(3,4,7,8,10,12,14,15)){
  Heart[[i]] <- factor(Heart[[i]])
}
# 变量类型修正后数据概况
skimr::skim(Heart)

##############################################################

# 数据拆分
set.seed(4321)
datasplit <- initial_split(Heart, prop = 0.75, strata = AHD)
traindata <- training(datasplit)
testdata <- testing(datasplit)

##############################################################

# 数据预处理
# 先对照训练集写配方
datarecipe <- recipe(AHD ~ ., traindata) %>%
  step_rm(Id) %>%
  step_naomit(all_predictors(), skip = F) %>%
  step_dummy(all_nominal_predictors()) %>%
  prep()
datarecipe

# 按方处理训练集和测试集
traindata2 <- bake(datarecipe, new_data = NULL) %>%
  dplyr::select(AHD, everything())
testdata2 <- bake(datarecipe, new_data = testdata) %>%
  dplyr::select(AHD, everything())

# 数据预处理后数据概况
skimr::skim(traindata2)
skimr::skim(testdata2)

##############################################################

# 训练模型
# 设定模型
model_logistic <- logistic_reg(
  mode = "classification",
  engine = "glm"
)
model_logistic

# 拟合模型
fit_logistic <- model_logistic %>%
  fit(AHD ~ ., traindata2)
fit_logistic
fit_logistic$fit
summary(fit_logistic$fit)

# 系数输出
fit_logistic %>%
  tidy()

##############################################################

# 应用模型-预测训练集
predtrain_logistic <- fit_logistic %>%
  predict(new_data = traindata2, type = "prob") %>%
  bind_cols(traindata2 %>% select(AHD)) %>%
  mutate(dataset = "train")
predtrain_logistic

# 评估模型ROC曲线-训练集上
levels(traindata2$AHD)
roctrain_logistic <- predtrain_logistic %>%
  roc_curve(AHD, .pred_No, event_level = "first") %>%
  mutate(dataset = "train")
roctrain_logistic
autoplot(roctrain_logistic)

# 约登法则对应的p值
yueden_logistic <- roctrain_logistic %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_logistic
# 预测概率+约登法则=预测分类
predtrain_logistic2 <- predtrain_logistic %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_logistic, "No", "Yes")))
predtrain_logistic2
# 混淆矩阵
cmtrain_logistic <- predtrain_logistic2 %>%
  conf_mat(truth = AHD, estimate = .pred_class)
cmtrain_logistic
autoplot(cmtrain_logistic, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))

# 合并指标
eval_train_logistic <- cmtrain_logistic %>%
  summary(event_level = "first") %>%
  bind_rows(predtrain_logistic %>%
              roc_auc(AHD, .pred_No, event_level = "first")) %>%
  mutate(dataset = "train")
eval_train_logistic

##############################################################

# 应用模型-预测测试集
predtest_logistic <- fit_logistic %>%
  predict(new_data = testdata2, type = "prob") %>%
  bind_cols(testdata2 %>% select(AHD)) %>%
  mutate(dataset = "test") %>%
  mutate(model = "logistic")
predtest_logistic

# 评估模型ROC曲线-测试集上
roctest_logistic <- predtest_logistic %>%
  roc_curve(AHD, .pred_No, event_level = "first") %>%
  mutate(dataset = "test")
autoplot(roctest_logistic)

# 预测概率+约登法则=预测分类
predtest_logistic2 <- predtest_logistic %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_logistic, "No", "Yes")))
predtest_logistic2

# 混淆矩阵
cmtest_logistic <- predtest_logistic2 %>%
  conf_mat(truth = AHD, estimate = .pred_class)
cmtest_logistic

autoplot(cmtest_logistic, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))

# 合并指标
eval_test_logistic <- cmtest_logistic %>%
  summary(event_level = "first") %>%
  bind_rows(predtest_logistic %>%
              roc_auc(AHD, .pred_No, event_level = "first")) %>%
  mutate(dataset = "test")
eval_test_logistic

##############################################################

# 合并训练集和测试集上ROC曲线
roctrain_logistic %>%
  bind_rows(roctest_logistic) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(linewidth = 1) +
  theme_bw()

# 合并训练集和测试集上性能指标
eval_logistic <- eval_train_logistic %>%
  bind_rows(eval_test_logistic) %>%
  mutate(model = "logistic")
eval_logistic

#################################################################
#################################################################



# 设定5折交叉验证
set.seed(42)
folds <- vfold_cv(traindata2, v = 5)
folds

# workflow
wf_logistic <- 
  workflow() %>%
  add_model(model_logistic) %>%
  add_formula(AHD ~ .)
wf_logistic

# 交叉验证
set.seed(42)
cv_logistic <- 
  wf_logistic %>%
  fit_resamples(folds,
                metrics = metric_set(yardstick::accuracy, 
                                     yardstick::roc_auc, 
                                     yardstick::pr_auc),
                control = control_resamples(save_pred = T))
cv_logistic

# 交叉验证指标平均结果
eval_cv_logistic <- collect_metrics(cv_logistic)
eval_cv_logistic


# 交叉验证指标具体结果
eval_cv5_logistic <- collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  roc_auc(AHD, .pred_No, event_level = "first") %>%
  ungroup() %>%
  mutate(model = "logistic") %>%
  left_join(eval_cv_logistic[c(1,3,5)])
eval_cv5_logistic

# 保存评估结果
save(fit_logistic,
     predtest_logistic,
     eval_logistic,
     eval_cv5_logistic, 
     file = ".\\cls2\\evalresult_logistic.RData")

# 交叉验证指标图示
eval_cv5_logistic %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = id, y = .estimate, group = 1)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "", y = "roc_auc") +
  theme_bw()

# 交叉验证图示
collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  roc_curve(AHD, .pred_No, event_level = "first") %>%
  ungroup() %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = id)) +
  geom_path(linewidth = 1) +
  theme_bw()


###################################################################

# 自变量数据集
colnames(traindata2)
traindatax <- traindata2[,-1]
colnames(traindatax)

# iml包
library(iml)
predictor_model <- Predictor$new(
  fit_logistic, 
  data = traindatax,
  y = traindata2$AHD,
  predict.function = function(model, newdata){
    predict(model, newdata, type = "prob") %>%
      pull(2)
  }
)

# 变量重要性-基于置换
imp_model <- FeatureImp$new(
  predictor_model, 
  loss = function(actual, predicted){
    return(1-Metrics::auc(as.numeric(actual=="Yes"), predicted))
  }
)
# 数值
imp_model$results
# 图示
imp_model$plot() +
  theme_bw()


# 变量效应
pdp_model <- FeatureEffect$new(
  predictor_model, 
  feature = "Age",
  method = "pdp"
)
# 数值
pdp_model$results
# 图示
pdp_model$plot() +
  theme_bw()

# 所有变量的效应全部输出
effs_model <- FeatureEffects$new(predictor_model, method = "pdp")
# 数值
effs_model$results
# 图示
effs_model$plot()

# 单样本shap分析
shap_model <- Shapley$new(
  predictor_model, 
  x.interest = traindatax[1,]
)
# 数值
shap_model$results
# 图示
shap_model$plot() +
  theme_bw()

# 基于所有样本的shap分析
# fastshap包
library(fastshap)
shap <- explain(
  fit_logistic, 
  X = as.data.frame(traindatax),
  nsim = 10,
  adjust = T,
  pred_wrapper = function(model, newdata) {
    predict(model, newdata, type = "prob") %>% pull(2)
  }
)

# 单样本图示
force_plot(object = shap[1L, ], 
           feature_values = as.data.frame(traindatax)[1L, ], 
           baseline = mean(predtrain_logistic$.pred_Yes), 
           display = "viewer") 

# 变量重要性
autoplot(shap, fill = "skyblue") +
  theme_bw()

data1 <- shap %>%
  as.data.frame() %>%
  dplyr::mutate(id = 1:n()) %>%
  pivot_longer(cols = -(ncol(traindatax)+1), values_to = "shap")
shapimp <- data1 %>%
  dplyr::group_by(name) %>%
  dplyr::summarise(shap.abs.mean = mean(abs(shap))) %>%
  dplyr::arrange(shap.abs.mean) %>%
  dplyr::mutate(name = forcats::as_factor(name))
data2 <- traindatax  %>%
  dplyr::mutate(id = 1:n()) %>%
  pivot_longer(cols = -(ncol(traindatax)+1))

# 所有变量shap图示
library(ggbeeswarm)
data1 %>%
  left_join(data2) %>%
  dplyr::rename("feature" = "name") %>%
  dplyr::group_by(feature) %>%
  dplyr::mutate(
    value = (value - min(value)) / (max(value) - min(value)),
    feature = factor(feature, levels = levels(shapimp$name))
  ) %>%
  dplyr::arrange(value) %>%
  dplyr::ungroup() %>%
  ggplot(aes(x = shap, y = feature, color = value)) +
  geom_quasirandom(width = 0.2) +
  scale_color_gradient(
    low = "red", 
    high = "blue", 
    breaks = c(0, 1), 
    labels = c(" Low", "High "), 
    guide = guide_colorbar(barwidth = 1, 
                           barheight = 20,
                           ticks = F,
                           title.position = "right",
                           title.hjust = 0.5)
  ) +
  labs(x = "SHAP value", color = "Feature value") +
  theme_bw() +
  theme(legend.title = element_text(angle = -90))



# 单变量shap图示
data1 %>%
  left_join(data2) %>%
  dplyr::rename("feature" = "name") %>%
  dplyr::filter(feature == "MaxHR") %>%
  ggplot(aes(x = value, y = shap)) +
  geom_point() +
  geom_smooth(se = F, span = 0.5) +
  labs(x = "MaxHR") +
  theme_bw()











# 模型机器---R语言tidymodels包机器学习分类与回归模型---二分类---xgboost

# https://www.tidymodels.org/find/parsnip/
# https://parsnip.tidymodels.org/reference/boost_tree.html
# https://parsnip.tidymodels.org/reference/details_boost_tree_xgboost.html

# 模型评估指标
# https://cran.r-project.org/web/packages/yardstick/vignettes/metric-types.html


library(tidymodels)

# 读取数据
Heart <- readr::read_csv(file.choose()) # tibble
colnames(Heart)
TP <- readr::read_csv(file.choose("D:/研一/研一下/利奈唑胺/机器学习new变量/newml.csv"))
colnames(TP)
EX <- readr::read_csv(file.choose("D:/研一/研一下/利奈唑胺/机器学习new变量/test72.csv"))
colnames(EX)

# 修正变量类型
# 将分类变量转换为factor

for(i in c(1,2,3,4,5,6,7,8,9,12,13,14,15,16,17,19,20,21,22,23,24)){
  TP[[i]] <- factor(TP[[i]])
}
for(i in c(1,2,3,4,5,6,7,8,9,12,13,14,15,16,17,19,20,21,22,23,24)){
  EX[[i]] <- factor(EX[[i]])
}
# 变量类型修正后数据概况
skimr::skim(Heart)
skimr::skim(TP)
skimr::skim(EX)

###############################################################

# 数据拆分
set.seed(4321)
datasplit <- initial_split(Heart, prop = 0.75, strata = AHD)
traindata <- training(datasplit)
testdata <- testing(datasplit)
traindata <- TP
testdata <- EX
###############################################################

# 数据预处理
# 先对照训练集写配方
datarecipe <- recipe(status ~ ., traindata) %>%
  step_naomit(all_predictors(), skip = F) %>%
  step_dummy(all_nominal_predictors()) %>%
  prep()
datarecipe

# 按方处理训练集和测试集
traindata2 <- bake(datarecipe, new_data = NULL) %>%
  dplyr::select(status, everything())
testdata2 <- bake(datarecipe, new_data = testdata) %>%
  dplyr::select(status, everything())

# 数据预处理后数据概况
skimr::skim(traindata2)
skimr::skim(testdata2)

###############################################################

# 训练模型
# 设定模型trees = 1000 stop_iter = 25 validation = 0.2
model_xgboost <- boost_tree(
  mode = "classification",
  engine = "xgboost",
  mtry = tune(),
  trees = 2000,
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = 0.01,
  loss_reduction = tune(),
  sample_size = tune(),
  stop_iter = 25
) %>%
  set_args(validation = 0.2)
model_xgboost

# workflow
wk_xgboost <- 
  workflow() %>%
  add_model(model_xgboost) %>%
  add_formula(status ~ .)
wk_xgboost

# 重抽样设定-5折交叉验证v = 5
set.seed(42)
folds <- vfold_cv(traindata2, v = 5)
folds

# 超参数寻优范围size = 5learn_rate(range = c(-3, -1))
hpset_xgboost <- parameters(
  mtry(range = c(1, 32)),
  min_n(range = c(2, 10)),
  tree_depth(range = c(2, 3)),
  loss_reduction(range = c(-12, 2)),
  sample_prop(range = c(0.8, 1))
)
# hpgrid_xgboost <- 
#   grid_regular(hpset_xgboost, levels = c(3, 2, 2, 3, 2, 2))
set.seed(42)
hpgrid_xgboost <- grid_random(hpset_xgboost, size = 6)
hpgrid_xgboost


# 交叉验证随机搜索过程
set.seed(42)
tune_xgboost <- wk_xgboost %>%
  tune_grid(resamples = folds,
            grid = hpgrid_xgboost,
            metrics = metric_set(yardstick::accuracy, 
                                 yardstick::roc_auc, 
                                 yardstick::pr_auc),
            control = control_grid(save_pred = T, verbose = T))

# 图示交叉验证结果
autoplot(tune_xgboost)
eval_tune_xgboost <- tune_xgboost %>%
  collect_metrics()
eval_tune_xgboost

# 经过交叉验证得到的最优超参数
hpbest_xgboost <- tune_xgboost %>%
  select_best(metric = "roc_auc")
hpbest_xgboost

# 采用最优超参数组合训练最终模型
set.seed(42)
final_xgboost <- wk_xgboost %>%
  finalize_workflow(hpbest_xgboost) %>%
  fit(traindata2)
final_xgboost

# 提取最终的算法模型
final_xgboost2 <- final_xgboost %>%
  extract_fit_engine()

# 变量重要性
importance_matrix <- xgb.importance(model = final_xgboost2)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix,
                    measure = "Cover",
                    col = "skyblue")
# SHAP
colnames(traindata2)
xgb.plot.shap(data = as.matrix(traindata2[,-1]), 
              model = final_xgboost2,
              top_n = 5)



###############################################################

# 应用模型-预测训练集
predtrain_xgboost <- final_xgboost %>%
  predict(new_data = traindata2, type = "prob") %>%
  bind_cols(traindata2 %>% select(status)) %>%
  mutate(dataset = "train")
predtrain_xgboost 
# 评估模型ROC曲线-训练集上
levels(traindata2$status)
roctrain_xgboost <- predtrain_xgboost %>%
  roc_curve(status, .pred_0, event_level = "first") %>%
  mutate(dataset = "train")
roctrain_xgboost
autoplot(roctrain_xgboost)

# 约登法则对应的p值
yueden_xgboost <- roctrain_xgboost %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_xgboost
# 预测概率+约登法则=预测分类
predtrain_xgboost2 <- predtrain_xgboost %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_0 >= yueden_xgboost, "0", "1")))
predtrain_xgboost2
# 混淆矩阵
cmtrain_xgboost <- predtrain_xgboost2 %>%
  conf_mat(truth = status, estimate = .pred_class)
cmtrain_xgboost
autoplot(cmtrain_xgboost, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
eval_train_xgboost <- cmtrain_xgboost %>%
  summary(event_level = "first") %>%
  bind_rows(predtrain_xgboost %>%
              roc_auc(status, .pred_0, event_level = "first")) %>%
  mutate(dataset = "train")
eval_train_xgboost

###############################################################

# 应用模型-预测测试集
predtest_xgboost <- final_xgboost %>%
  predict(new_data = testdata2, type = "prob") %>%
  bind_cols(testdata2 %>% select(status)) %>%
  mutate(dataset = "test") %>%
  mutate(model = "xgboost")
predtest_xgboost

# 评估模型ROC曲线-测试集上
roctest_xgboost <- predtest_xgboost %>%
  roc_curve(status, .pred_0, event_level = "first") %>%
  mutate(dataset = "test")
roctest_xgboost
autoplot(roctest_xgboost)


# 预测概率+约登法则=预测分类
predtest_xgboost2 <- predtest_xgboost %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_0 >= yueden_xgboost, "0", "1")))
predtest_xgboost2
# 混淆矩阵
cmtest_xgboost <- predtest_xgboost2 %>%
  conf_mat(truth = status, estimate = .pred_class)
cmtest_xgboost
autoplot(cmtest_xgboost, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
eval_test_xgboost <- cmtest_xgboost %>%
  summary(event_level = "first") %>%
  bind_rows(predtest_xgboost %>%
              roc_auc(status, .pred_0, event_level = "first")) %>%
  mutate(dataset = "test")
eval_test_xgboost

###############################################################


# 合并训练集和测试集上ROC曲线
roctrain_xgboost %>%
  bind_rows(roctest_xgboost) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(linewidth = 1) +
  theme_bw()

# 合并训练集和测试集上性能指标
eval_xgboost <- eval_train_xgboost %>%
  bind_rows(eval_test_xgboost) %>%
  mutate(model = "xgboost")
eval_xgboost

#############################################################

# 最优超参数的交叉验证指标平均结果
eval_best_cv_xgboost <- eval_tune_xgboost %>%
  inner_join(hpbest_xgboost[, 1:6])
eval_best_cv_xgboost

# 最优超参数的交叉验证指标具体结果
eval_best_cv5_xgboost <- tune_xgboost %>%
  collect_predictions() %>%
  inner_join(hpbest_xgboost[, 1:6]) %>%
  group_by(id) %>%
  roc_auc(AHD, .pred_No) %>%
  ungroup() %>%
  mutate(model = "xgboost") %>%
  inner_join(eval_best_cv_xgboost[c(7,9,11)])
eval_best_cv5_xgboost

# 保存评估结果
save(final_xgboost,
     predtest_xgboost,
     eval_xgboost,
     eval_best_cv5_xgboost, 
     file = ".\\cls2\\evalresult_xgboost.RData")

# 最优超参数的交叉验证指标图示
eval_best_cv5_xgboost %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = id, y = .estimate, group = 1)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "", y = "roc_auc") +
  theme_bw()

# 最优超参数的交叉验证图示
tune_xgboost %>%
  collect_predictions() %>%
  inner_join(hpbest_xgboost[, 1:6]) %>%
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
  final_xgboost, 
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
  final_xgboost, 
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
           baseline = mean(predtrain_xgboost$.pred_Yes), 
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











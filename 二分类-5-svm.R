# 模型机器---R语言tidymodels包机器学习分类与回归模型---二分类---SVM

# https://www.tidymodels.org/find/parsnip/
# https://parsnip.tidymodels.org/reference/svm_rbf.html
# https://parsnip.tidymodels.org/reference/details_svm_rbf_kernlab.html

# 模型评估指标
# https://cran.r-project.org/web/packages/yardstick/vignettes/metric-types.html

library(tidymodels)

# 读取数据
Heart <- readr::read_csv(file.choose())
colnames(Heart)

# 修正变量类型
# 将分类变量转换为factor
for(i in c(3,4,7,8,10,12,14,15)){
  Heart[[i]] <- factor(Heart[[i]])
}
# 变量类型修正后数据概况
skimr::skim(Heart)

###############################################################

# 数据拆分
set.seed(4321)
datasplit <- initial_split(Heart, prop = 0.75, strata = AHD)
traindata <- training(datasplit)
testdata <- testing(datasplit)

###############################################################

# 数据预处理
# 先对照训练集写配方
datarecipe <- recipe(AHD ~ ., traindata) %>%
  step_rm(Id) %>%
  step_naomit(all_predictors(), skip = F) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors()) %>%
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

###############################################################

# 训练模型
# 设定模型
model_rsvm <- svm_rbf(
  mode = "classification",
  engine = "kernlab",
  cost = tune(),
  rbf_sigma = tune()
)
model_rsvm

# workflow
wk_rsvm <- 
  workflow() %>%
  add_model(model_rsvm) %>%
  add_formula(AHD ~ .)
wk_rsvm

# 重抽样设定-5折交叉验证
set.seed(42)
folds <- vfold_cv(traindata2, v = 5)
folds

# 超参数寻优范围
hpset_rsvm <- parameters(cost(range = c(-5, 5)), 
                    rbf_sigma(range = c(-4, -1)))
hpgrid_rsvm <- grid_regular(hpset_rsvm, levels = c(2,3))
hpgrid_rsvm


# 交叉验证网格搜索过程
set.seed(42)
tune_rsvm <- wk_rsvm %>%
  tune_grid(resamples = folds,
            grid = hpgrid_rsvm,
            metrics = metric_set(yardstick::accuracy, 
                                 yardstick::roc_auc, 
                                 yardstick::pr_auc),
            control = control_grid(save_pred = T, verbose = T))

# 图示交叉验证结果
autoplot(tune_rsvm)
eval_tune_rsvm <- tune_rsvm %>%
  collect_metrics()
eval_tune_rsvm


# 经过交叉验证得到的最优超参数
hpbest_rsvm <- tune_rsvm %>%
  select_best(metric = "roc_auc")
hpbest_rsvm

# 采用最优超参数组合训练最终模型
final_rsvm <- wk_rsvm %>%
  finalize_workflow(hpbest_rsvm) %>%
  fit(traindata2)
final_rsvm

# 提取最终的算法模型
final_rsvm %>%
  extract_fit_engine()

###############################################################

# 应用模型-预测训练集
predtrain_rsvm <- final_rsvm %>%
  predict(new_data = traindata2, type = "prob") %>%
  bind_cols(traindata2 %>% select(AHD)) %>%
  mutate(dataset = "train")
predtrain_rsvm
# 评估模型ROC曲线-训练集上
levels(traindata2$AHD)
roctrain_rsvm <- predtrain_rsvm %>%
  roc_curve(AHD, .pred_No, event_level = "first") %>%
  mutate(dataset = "train")
roctrain_rsvm
autoplot(roctrain_rsvm)

# 约登法则对应的p值
yueden_rsvm <- roctrain_rsvm %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_rsvm
# 预测概率+约登法则=预测分类
predtrain_rsvm2 <- predtrain_rsvm %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_rsvm, "No", "Yes")))
predtrain_rsvm2
# 混淆矩阵
cmtrain_rsvm <- predtrain_rsvm2 %>%
  conf_mat(truth = AHD, estimate = .pred_class)
cmtrain_rsvm
autoplot(cmtrain_rsvm, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
eval_train_rsvm <- cmtrain_rsvm %>%
  summary(event_level = "first") %>%
  bind_rows(predtrain_rsvm %>%
              roc_auc(AHD, .pred_No, event_level = "first")) %>%
  mutate(dataset = "train")
eval_train_rsvm

###############################################################


# 应用模型-预测测试集
predtest_rsvm <- final_rsvm %>%
  predict(new_data = testdata2, type = "prob") %>%
  bind_cols(testdata2 %>% select(AHD)) %>%
  mutate(dataset = "test") %>%
  mutate(model = "rsvm")
predtest_rsvm
# 评估模型ROC曲线-测试集上
roctest_rsvm <- predtest_rsvm %>%
  roc_curve(AHD, .pred_No, event_level = "first") %>%
  mutate(dataset = "test")
roctest_rsvm
autoplot(roctest_rsvm)


# 预测概率+约登法则=预测分类
predtest_rsvm2 <- predtest_rsvm %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_rsvm, "No", "Yes")))
predtest_rsvm2
# 混淆矩阵
cmtest_rsvm <- predtest_rsvm2 %>%
  conf_mat(truth = AHD, estimate = .pred_class)
cmtest_rsvm
autoplot(cmtest_rsvm, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
eval_test_rsvm <- cmtest_rsvm %>%
  summary(event_level = "first") %>%
  bind_rows(predtest_rsvm %>%
              roc_auc(AHD, .pred_No, event_level = "first")) %>%
  mutate(dataset = "test")
eval_test_rsvm
###############################################################


# 合并训练集和测试集上ROC曲线
roctrain_rsvm %>%
  bind_rows(roctest_rsvm) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(linewidth = 1) +
  theme_bw()

# 合并训练集和测试集上性能指标
eval_rsvm <- eval_train_rsvm %>%
  bind_rows(eval_test_rsvm) %>%
  mutate(model = "rsvm")
eval_rsvm

#############################################################

# 最优超参数的交叉验证指标平均结果
eval_best_cv_rsvm <- eval_tune_rsvm %>%
  inner_join(hpbest_rsvm[, 1:2])
eval_best_cv_rsvm

# 最优超参数的交叉验证指标具体结果
eval_best_cv5_rsvm <- tune_rsvm %>%
  collect_predictions() %>%
  inner_join(hpbest_rsvm[, 1:2]) %>%
  group_by(id) %>%
  roc_auc(AHD, .pred_No) %>%
  ungroup() %>%
  mutate(model = "rsvm") %>%
  inner_join(eval_best_cv_rsvm[c(3,5,7)])
eval_best_cv5_rsvm

# 保存评估结果
save(final_rsvm,
     predtest_rsvm,
     eval_rsvm,
     eval_best_cv5_rsvm, 
     file = ".\\cls2\\evalresult_rsvm.RData")

# 最优超参数的交叉验证指标图示
eval_best_cv5_rsvm %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = id, y = .estimate, group = 1)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "", y = "roc_auc") +
  theme_bw()

# 最优超参数的交叉验证图示
tune_rsvm %>%
  collect_predictions() %>%
  inner_join(hpbest_rsvm[, 1:2]) %>%
  group_by(id) %>%
  roc_curve(AHD, .pred_No) %>%
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
  final_rsvm, 
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
  final_rsvm, 
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
           baseline = mean(predtrain_rsvm$.pred_Yes), 
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

















#############################################################

# 线性核svm
# https://www.tidymodels.org/find/parsnip/
# https://parsnip.tidymodels.org/reference/svm_linear.html
# https://parsnip.tidymodels.org/reference/details_svm_linear_kernlab.html
model_lsvm <- svm_linear(
  mode = "classification",
  engine = "kernlab",
  cost = tune()
)
hpset_lsvm <- parameters(cost(range = c(-5, 5)))
hpgrid_lsvm <- grid_regular(hpset_lsvm, levels = 5)
hpgrid_lsvm


# 多项式核SVM
# https://www.tidymodels.org/find/parsnip/
# https://parsnip.tidymodels.org/reference/svm_poly.html
# https://parsnip.tidymodels.org/reference/details_svm_poly_kernlab.html
model_psvm <- svm_poly(
  mode = "classification",
  engine = "kernlab",
  cost = tune(),
  degree = tune(),
  scale_factor = tune()
)
hpset_psvm <- parameters(cost(range = c(-5, 5)),
                    degree(),
                    scale_factor(range = c(-5, -1)))
hpgrid_psvm <- grid_regular(hpset_psvm, levels = c(3, 2, 2))
hpgrid_psvm


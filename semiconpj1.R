library(readxl)
data <- read_excel("data_summary_report.xlsx")
data <- data[ , -c(1, 2, 3, 5, 19)] #remove index, processID, timestamp, dayofweek, waferID

data$Defect <- as.factor(data$Defect)
data$Tool_Type = as.factor(data$Tool_Type)
data$Join_Status =as.factor(data$Join_Status)
summary(data)
attach(data)


#multi-dimensional scaling (MDS)
library(stats)
library(ggplot2)
set.seed(123)
data_sample <- data[sample(nrow(data), 1000), ] #สุ่มดึงข้อมูลมา 1,000 แถว
#รัน dist กับข้อมูลที่เล็กลง
dist1=dist(data_sample[,2:11], method='euclidean') #ใช้คอลัมน์ที่เป็น Sensor Data
wafer.mds<- cmdscale(dist1, k=2, eig=TRUE) #(k=2 คือลดเหลือ 2 มิติ)
#สร้าง Data Frame ใหม่สำหรับ Plot (รวมค่า MDS และ Label เข้าด้วยกัน)
mds_plot_data <- data.frame(
  MDS1 = wafer.mds$points[, 1], #x axis mdsx
  MDS2 = wafer.mds$points[, 2], #y axis mdsy
  Defect = factor(data_sample$Defect) # ใช้ Defect เป็นตัวแบ่งสีและรูปร่าง
)
p=ggplot(mds_plot_data, aes(x = MDS1, y = MDS2, colour = Defect, shape = Defect)) #use Defect to seperate group
p+geom_point(size = 3, alpha = 0.7) +
  theme_minimal() +
  labs(title = "MDS for Wafer Semiconductor Data") +
  scale_color_manual(values = c("blue", "red")) # 0 = Blue, 1 = Red
#check characteristic of these 2 groups from 2 dimensions

# ลอง standadize data Score (Mean=0, SD=1)
sam_scaled <- scale(data_sample[, 2:11])
dist2 <- dist(sam_scaled, method = 'euclidean')
wafer.mds2<- cmdscale(dist2, k=2, eig=TRUE)
mds_plot_data2 <- data.frame(
  MDSX = wafer.mds2$points[, 1], 
  MDSY = wafer.mds2$points[, 2], 
  Defect = factor(data_sample$Defect) # ดึง Defect มาจาก data_sample แทน sam_scaled นจ scaled เป็น vector
)
q=ggplot(mds_plot_data2, aes(x = MDSX, y = MDSY, colour = Defect, shape = Defect))
q+geom_point(size = 3, alpha = 0.7) +
  theme_minimal() +
  labs(title = "MDS with Scaled Data") +
  scale_color_manual(values = c("blue", "red")) # 0 = Blue, 1 = Red


###outlier test ? >> use model that can endure outlier as some outlier is signal of non-join wafer

wf_scaled <- scale(data[, 2:11]) #normalize all data




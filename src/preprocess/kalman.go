package preprocess

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
)

// KalmanFilter 简单的1D卡尔曼滤波器（位置-速度模型）
type KalmanFilter struct {
	x [2]float64    // 状态向量 [位置, 速度]
	P [2][2]float64 // 协方差矩阵
	Q [2][2]float64 // 过程噪声
	R float64       // 测量噪声
}

// NewKalmanFilter 初始化卡尔曼滤波器
func NewKalmanFilter(initialPos, initialVel, processNoise, measurementNoise float64) *KalmanFilter {
	kf := &KalmanFilter{
		x: [2]float64{initialPos, initialVel},
		P: [2][2]float64{{1, 0}, {0, 1}}, // 初始协方差
		Q: [2][2]float64{{processNoise, 0}, {0, processNoise}},
		R: measurementNoise,
	}
	return kf
}

// Predict 预测步骤
func (kf *KalmanFilter) Predict(dt float64) {
	// 状态转移矩阵 F = [[1, dt], [0, 1]]
	kf.x[0] += kf.x[1] * dt // 位置 += 速度 * dt
	// 速度不变（假设恒定速度）

	// 协方差预测 P = F*P*F^T + Q
	F := [2][2]float64{{1, dt}, {0, 1}}
	P_new := [2][2]float64{}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			for k := 0; k < 2; k++ {
				for l := 0; l < 2; l++ {
					P_new[i][j] += F[i][k] * kf.P[k][l] * F[j][l]
				}
			}
			P_new[i][j] += kf.Q[i][j]
		}
	}
	kf.P = P_new
}

// Update 更新步骤
func (kf *KalmanFilter) Update(measurement float64) {
	// 测量矩阵 H = [1, 0]
	y := measurement - kf.x[0] // 残差

	// 卡尔曼增益 K = P*H^T / (H*P*H^T + R)
	S := kf.P[0][0] + kf.R // H*P*H^T = P[0][0]
	K := [2]float64{kf.P[0][0] / S, kf.P[1][0] / S}

	// 更新状态 x = x + K*y
	kf.x[0] += K[0] * y
	kf.x[1] += K[1] * y

	// 更新协方差 P = (I - K*H)*P
	I_KH := [2][2]float64{{1 - K[0], 0}, {-K[1], 1}}
	P_new := [2][2]float64{}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			for k := 0; k < 2; k++ {
				P_new[i][j] += I_KH[i][k] * kf.P[k][j]
			}
		}
	}
	kf.P = P_new
}

// GetPosition 获取当前位置
func (kf *KalmanFilter) GetPosition() float64 {
	return kf.x[0]
}

func main() {
	// 命令行参数
	inputFile := flag.String("input", "data/track1.csv", "输入CSV文件路径")
	flag.Parse()

	// 打开文件
	file, err := os.Open(*inputFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	// 读取CSV
	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	if len(records) < 2 {
		log.Fatal("CSV文件至少需要标题行和一行数据")
	}

	// 假设第一行是标题：t,x,y,z
	// 初始化滤波器
	var kfX, kfY, kfZ *KalmanFilter
	var prevT float64

	for i, record := range records {
		if i == 0 {
			// 标题行，跳过
			continue
		}

		if len(record) < 4 {
			log.Printf("第%d行数据不完整，跳过", i+1)
			continue
		}

		t, err1 := strconv.ParseFloat(record[0], 64)
		x, err2 := strconv.ParseFloat(record[1], 64)
		y, err3 := strconv.ParseFloat(record[2], 64)
		z, err4 := strconv.ParseFloat(record[3], 64)

		if err1 != nil || err2 != nil || err3 != nil || err4 != nil {
			log.Printf("第%d行解析错误，跳过", i+1)
			continue
		}

		if i == 1 {
			// 初始化滤波器
			kfX = NewKalmanFilter(x, 0, 0.01, 0.1) // 初始速度0，过程噪声0.01，测量噪声0.1
			kfY = NewKalmanFilter(y, 0, 0.01, 0.1)
			kfZ = NewKalmanFilter(z, 0, 0.01, 0.1)
			prevT = t
			// 输出初始值
			fmt.Printf("%.3f,%.3f,%.3f,%.3f\n", t, x, y, z)
			continue
		}

		dt := t - prevT
		prevT = t

		// 预测
		kfX.Predict(dt)
		kfY.Predict(dt)
		kfZ.Predict(dt)

		// 更新
		kfX.Update(x)
		kfY.Update(y)
		kfZ.Update(z)

		// 输出滤波后的值
		fmt.Printf("%.3f,%.3f,%.3f,%.3f\n", t, kfX.GetPosition(), kfY.GetPosition(), kfZ.GetPosition())
	}
}

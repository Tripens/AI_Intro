import random

class Perceptron:
    def __init__(self, input_size):
        self.weights = [random.uniform(-1, 1) for _ in range(input_size)]   #随机初始化权重
        self.bias = random.uniform(-1, 1)   #随机初始化偏置
        self.output = None
        self.downstream = []    #下游连接

    def activate(self, inputs):
        """计算感知器输出并传递给下游感知器"""
        total = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        self.output = 1 if total >= 0 else 0  # 阶跃激活函数
        
        # 将输出传递给下游感知器
        for perceptron in self.downstream:
            perceptron.receive_input(self.output)
        
        return self.output

    def receive_input(self, input_value):
        """接收来自上游感知器的输入"""
        # 此处需要配合PerceptronLayer使用
        pass

    def connect(self, downstream_perceptron):
        """连接到下游感知器"""
        self.downstream.append(downstream_perceptron)

class PerceptronLayer:
    def __init__(self, num_units, input_size):
        self.perceptrons = [Perceptron(input_size) for _ in range(num_units)]
        self.outputs = []
        self.expected_inputs = num_units  # 仅用于输出层

    def receive_input(self, input_value):
        """收集输入并触发计算（用于隐藏层/输出层）"""
        self.outputs.append(input_value)
        
        if len(self.outputs) == self.expected_inputs:
            return [p.activate(self.outputs) for p in self.perceptrons]
        return None

    def forward(self, inputs):
        """直接处理输入（用于输入层）"""
        return [p.activate(inputs) for p in self.perceptrons]

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        # 创建各层并建立连接
        for i in range(len(layer_sizes)-1):
            layer = PerceptronLayer(layer_sizes[i+1], layer_sizes[i])
            self.layers.append(layer)
            
            # 连接前一层到当前层（如果是第一层则跳过）
            if i > 0:
                for p in self.layers[i-1].perceptrons:
                    p.connect(layer)

    def forward(self, inputs):
        """前向传播"""
        current_outputs = self.layers[0].forward(inputs)
        for layer in self.layers[1:]:
            current_outputs = layer.receive_input(current_outputs)
        return current_outputs

# 示例用法
if __name__ == "__main__":
    # 创建网络结构：输入层2个节点，隐藏层3个节点，输出层1个节点
    network = NeuralNetwork([2, 3, 1])
    
    # 输入样本
    inputs = [0.5, 0.3]
    
    # 前向传播
    output = network.forward(inputs)
    print("Network output:", output)
    
    # 验证单个感知器连接
    p1 = Perceptron(2)
    p2 = Perceptron(1)
    p1.connect(p2)
    
    p1_output = p1.activate([0.5, 0.3])
    print("Perceptron 1 output:", p1_output)
    print("Perceptron 2 output:", p2.output)
## 1. 简化用例：
### 1.1. 假设：
- 一个权重矩阵*W*（16*16），torch.size([16,16])，精度：FP16
- VPTQ参数设置：
	- 量化向量长度：8
	- 主码本大小：16
	- 残差码本大小：4

### 1.2. 步骤
#### 1.2.1.原权重矩阵W:  (16\*16)
![[./W_org.png|600]]
原权重矩阵比特数：16\*16\*16bits  (16行，16列，一个权重16bits)

#### 1.2.2. 切分成长度为8的向量：
![[W_v.png|600]]
#### 1.2.3. K-means聚类（使用近似hessian矩阵加权），产生16个质心，质心数=主码本大小
![[main_codebook.png]]
左侧为原权重矩阵W，右侧即为生成的主码本，主码本条目由同颜色的原权重向量加权平均生成
主码本比特数：16\*8\*16bits  (16个质心，质心向量长度为8，一个权重16bits)
#### 1.2.4. 还需要index唯一标识主码本每个质心：
一共16个质心，需要`log(16) = 4`比特
![[index.png|300]]
右侧为主码本的index，省略了
#### 1.2.5. 为进一步减少误差，构造残差码本
将原权重矩阵W中的向量替换为主码本中对应类的质心向量（同颜色），构造出**主码本权重矩阵$W_{codebook}$** ， **残差矩阵$W_{res}$** = $W$ - $W_{codebook}$ （逐元素相减）
#### 1.2.6. 使用上述相同方法构造残差码本
区别在于残差码本质心数为4，因此index需要`log(4) = 2`比特
#### 1.2.7. 原权重矩阵中的一个向量被量化为主码本的一个质心与残差码本的一个质心之和
因此原来的一个向量可由两个码本的索引代表：总索引比特数 = 4+2 bits

### 1.3. 粗略计算
- 原有的一个权重向量为8\*16 bits由6 bits代表，量化比特数为6\/8 = 0.75bit
- 然而这个大小是未包含两个码本，只有index的
### 1.4. 包含码本的计算
- 量化前总比特数：16\*16\*16 bits
- 量化后总比特数： 
	- 向量个数：16\*16\/8 = 32
	- 质心index比特数： 4 + 2 = 6 bit
	- index表总比特数： 32  \* 6 = 192 bit  (一个向量对应一个index)
	- 主码本总比特数： 16\*8\*16 = 2048 bits  (16个质心，质心向量长度为8，一个权重16bits)
	- 残差码本总比特数： 4 \*8\*16 = 512 bits (4个质心)
	- 所以，最终总比特数为： 192 + 2048 + 512 = 2752 bits
	- 量化比特数为： 2752 / （16 \* 16）= 10.75 bit
假设的数值是随便设置的，所以与3bit相差很远，接下来是真实的计算。

## 2. 真实计算
### 2.1. LLaMA2-7B
#### 2.1.1. 参数设置
- 权重矩阵：
	- 权重精度： FP16
	- 模型的结构：

			所有参数名称和形状:
			embed_tokens.weight: torch.Size([32000, 4096])
			layers.0.self_attn.q_proj.weight: torch.Size([4096, 4096])
			layers.0.self_attn.k_proj.weight: torch.Size([4096, 4096])
			layers.0.self_attn.v_proj.weight: torch.Size([4096, 4096])
			layers.0.self_attn.o_proj.weight: torch.Size([4096, 4096])
			layers.0.mlp.gate_proj.weight: torch.Size([11008, 4096])
			layers.0.mlp.up_proj.weight: torch.Size([11008, 4096])
			layers.0.mlp.down_proj.weight: torch.Size([4096, 11008])
			layers.0.input_layernorm.weight: torch.Size([4096])
			layers.0.post_attention_layernorm.weight: torch.Size([4096])
			layers.1.self_attn.q_proj.weight: torch.Size([4096, 4096])
			layers.1.self_attn.k_proj.weight: torch.Size([4096, 4096])
			layers.1.self_attn.v_proj.weight: torch.Size([4096, 4096])
			layers.1.self_attn.o_proj.weight: torch.Size([4096, 4096])
			layers.1.mlp.gate_proj.weight: torch.Size([11008, 4096])
			layers.1.mlp.up_proj.weight: torch.Size([11008, 4096])
			layers.1.mlp.down_proj.weight: torch.Size([4096, 11008])
			layers.1.input_layernorm.weight: torch.Size([4096])
			layers.1.post_attention_layernorm.weight: torch.Size([4096])
			layers.2.self_attn.q_proj.weight: torch.Size([4096, 4096])
			layers.2.self_attn.k_proj.weight: torch.Size([4096, 4096])
VPTQ只对q,k,v,o,gate,up,down层进行量化，模型一共32层（0~31）
- VPTQ参数设置：对于每层的q,k,v,o,gate,up,down中的每一个权重矩阵
	- 向量长度：8
	- 主码本大小： 65536 个质心
	- 残差码本大小： 256 个质心
#### 2.1.2. 计算
##### 2.1.2.1. q矩阵
- [4096 , 4096]
- index长度 ： log(65536) + log(256) = 16 + 8 = 24 bits
- 向量个数 :  4096\* 4096 \/ 8 = 2097152 
- index表总比特数： 4096\* 4096 \/ 8 \*24 = 4096\* 4096 \*3
- 主码本总比特数： 65536 \* 8 \* 16 bit = 8388608 bit
- 残差码总本比特数： 256 \* 8 \* 16 bit = 32768 bit
- q矩阵总比特数： 4096\* 4096 \*3 + 8421376 
##### 2.1.2.2. k, v, o矩阵同理
##### 2.1.2.3. gate矩阵
- [11008 , 4096]
- index长度 ： log(65536) + log(256) = 16 + 8 = 24 bits
- 向量个数 :  11008\* 4096 \/ 8 = 5,636,096
-  index表总比特数： 11008\* 4096 \/ 8 \*24 = 11008\* 4096 \*3
- 主码本总比特数： 65536 \* 8 \* 16 bit = 8388608 bit
- 残差码总本比特数： 256 \* 8 \* 16 bit = 32768 bit
- gate矩阵总比特数： 11008\* 4096 \*3 + 8421376 
##### 2.1.2.4. up, down矩阵同理
##### 2.1.2.5. 总比特数
- （4096\* 4096 \*3 + 8421376 ）\* 4 + (11008\* 4096 \*3 + 8421376 ) \*3
- = 4096 \* (4096\*12+11008\*9) + 8421376 \* 7
- = 4096 \* (49152 + 99,072) + 58949632
- = 4096 \* 148,224 + 58949632
- = 607,125,504 + 58949632
- = 666,075,136

##### 2.1.2.6. 量化比特数
- 666,075,136 / （4096\* 4096 \*4 + 11008\* 4096 \*3） 
（一共7层：q,k,v,o,gate,up,down）
- = 666,075,136 / （16777216 \* 4 + 45,088,768 \*3）
- = 666,075,136 /  ( 67,108,864 + 135,266,304)
- = 666,075,136 /  202,375,168
- = 3.291288860103627 bit

### 2.2.  使用python脚本计算器计算
### 2.3. OPT
- 实际量化比特数 (Effective Bitwidth): 10.1389 bits

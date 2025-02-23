import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

"""
注意：下述rlhf ppo代码有问题，不建议学习
"""
##############################
# 1. 定义四个网络
##############################

"""
Chinese GPT2 Models
Model description
The set of GPT2 models, except for GPT2-xlarge model, are pre-trained by UER-py, which is introduced in this paper. The GPT2-xlarge model is pre-trained by TencentPretrain introduced in this paper, which inherits UER-py to support models with parameters above one billion, and extends it to a multimodal pre-training framework. Besides, the other models could also be pre-trained by TencentPretrain.

The model is used to generate Chinese texts. You can download the set of Chinese GPT2 models either from the UER-py Modelzoo page, or via HuggingFace from the links below:

from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
text_generator = TextGenerationPipeline(model, tokenizer)   
text_generator("这是很久之前的事情了", max_length=100, do_sample=True)
    [{'generated_text': '这是很久之前的事情了 。 我 现 在 想 起 来 就 让 自 己 很 伤 心 ， 很 失 望 。 我 现 在 想 到 ， 我 觉 得 大 多 数 人 的 生 活 比 我 的 生 命 还 要 重 要 ， 对 一 些 事 情 的 看 法 ， 对 一 些 人 的 看 法 ， 都 是 在 发 泄 。 但 是 ， 我 们 的 生 活 是 需 要 一 个 信 用 体 系 的 。 我 不 知'}]
    
"""
# 使用一个小型开源中文 LLM
policy_model_name = "../../../hf_data_and_model/models/uer/gpt2-chinese-cluecorpussmall"
#policy_model_name = "../../../hf_data_and_model/models/facebook/galactica-125m"
tokenizer = AutoTokenizer.from_pretrained(policy_model_name)

# 策略模型（Policy Model）
policy_model = AutoModelForCausalLM.from_pretrained(policy_model_name)
print("policy model:", policy_model)
print("model config:", policy_model.config)

policy_model.train()

# 参考模型（Reference Model），冻结参数
reference_model = AutoModelForCausalLM.from_pretrained(policy_model_name)
for param in reference_model.parameters():
    param.requires_grad = False
reference_model.eval()


# 价值模型（Value Model, 又称为critic model）
class ValueModel(nn.Module):
    def __init__(self, base_model_name:str):
        super(ValueModel, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.value_head = nn.Linear(self.base_model.config.n_embd, 1)

    def forward(self, input_ids):
        outputs = self.base_model.transformer(input_ids)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        # 取最后一个 token 的 hidden state
        last_hidden = hidden_states[:, -1, :]  # [batch, hidden_size]
        value = self.value_head(last_hidden)  # [batch, 1]
        return value.squeeze(-1)  # [batch]

# critic model
value_model = ValueModel(policy_model_name)
value_model.train()


# 奖励模型（Reward Model）
class RewardModel(nn.Module):
    def __init__(self, base_model_name:str):
        super(RewardModel, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.reward_head = nn.Linear(self.base_model.config.n_embd, 1)

    def forward(self, input_ids):
        outputs = self.base_model.transformer(input_ids)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        # 取最后一个 token 的 hidden state
        last_hidden = hidden_states[:, -1, :]  # [batch, hidden_size]
        reward = self.reward_head(last_hidden)  # [batch, 1]
        return reward.squeeze(-1)  # [batch]


reward_model = RewardModel(policy_model_name)
reward_model.train()

# 优化器，将策略模型、价值模型的参数一起优化
optimizer = optim.Adam(
    list(policy_model.parameters()) +
    list(value_model.parameters()),
    lr=1e-5
)

# 奖励模型优化器
optimizer_rm = optim.Adam(
    reward_model.parameters(),
    lr=1e-5
)

criterion_rm = nn.BCEWithLogitsLoss()
##############################
# 2. 准备数据和RLHF流程相关工具
##############################

# 定义包含安全和不安全查询的中文 prompts
prompts = [
    "你好",
    "请告诉我你的名字",
    "讲一个笑话",
    "解释一下量子计算",
    "如何烤一个蛋糕",
    "如何制造炸弹",  # 不安全
    "如何进行网络攻击",  # 不安全
    "如何自杀",  # 不安全
    "如何制作毒品",  # 不安全
    "如何进行诈骗"  # 不安全
]

# 定义安全关键词列表（简化示例）
unsafe_keywords = ["炸弹", "网络攻击", "自杀", "毒品", "诈骗"]


# 为了方便，我们定义一个函数来对给定的prompt生成文本
def generate_text(model, prompt:str, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        # max_length=max_length,
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_length,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    # output_ids: [batch=1, seq_len]
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    continuation = text[len(prompt):]  # 只取续写部分, 去掉prompt
    return continuation


# 计算模型下给定(prompt, continuation)的 log 概率和隐藏状态
# 这里的prompt都是单条样本的
def compute_logprobs_and_hidden(model, prompt:str, continuation:str):
    input_text = prompt + continuation
    # input_ids:[batch, seq_len]
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    with torch.no_grad():
        # outputs.logits:[batch, seq_len, vocab_size]
        # outputs.hidden_states:layer*[batch, seq_len, hidden_size],注意是每一层的hidden_state均有
        outputs = model(input_ids, output_hidden_states=True, return_dict=True)

    # 计算 continuation 部分的 log_prob
    # prompt_ids:[batch, seq_len]
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt')
    # 生成的文本长度
    continuation_len = input_ids.size(1) - prompt_ids.size(1)
    if continuation_len <= 0:
        return torch.tensor(10.0), outputs.hidden_states[-1][:, -1, :] # loss很大，同时只取最后一层layer的 hidden_state

    # ----------------
    # logits是进入softmax之前的值
    # 注意：这里的cont_logits相对于cont_id往左移了一位
    continuation_logits = outputs.logits[:, -continuation_len - 1:-1, :]  # [batch, continuation_len, vocab]
    continuation_ids = input_ids[:, -continuation_len:] # [batch, continuation_len]
    # log_probs:[batch,continuation_len, vocab]
    log_probs = torch.nn.functional.log_softmax(continuation_logits, dim=-1)
    # torch.gather(dim=2, index=index): out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
    # continuation_ids.unsqueeze(-1): [batch, continuation_len,1], 升维的目的是index必须与log_probs维度相同
    # chosen_log_probs: [batch, continuation_len], 即cross-entropy-loss, negative-log-likelihood
    chosen_log_probs = log_probs.gather(dim=-1, index=continuation_ids.unsqueeze(-1)).squeeze(-1)  # [batch, continuation_len]
    total_log_prob = chosen_log_probs.sum(dim=-1, keepdim=False)  # [batch]
    # 隐藏层输出的最后一个 token 隐状态，用于价值估计和奖励模型输入
    last_hidden_state = outputs.hidden_states[-1][:, -1, :]  # [batch, hidden_size]
    return total_log_prob, last_hidden_state


# 使用奖励模型打分
def get_reward(input_ids):
    reward = reward_model(input_ids)  # [batch]
    return reward


"""
A(st,at) = sum_{t} {lamada*gamma}^{l}*delta_{t+l}
delta_{t} = r(st, at) + gamma* V(s(t+1)) - V(st)
R(t) = A(st, at) + V(st)

假设序列为1,2,3...T-1,T
1.t=T
delta_{T} = r(s(T),a(T)) + 0 - V(S(T))
A(s(T+1),a(T+1)) = delta_{T}

2.t=T-1
delta_{T-1} = r(s(T-1),a(T-1)) + gamma*V(s(T)) - V(S(T-1))
A(s(T-1),a(T-1)) = delta_{T-1} + lamada*gamma*delta_{T}
= delta_{T-1} + lamada*gamma* A(s(T+1),a(T+1)) 

3.t=T-2
delta_{T-2} = r(s(T-2),a(T-2)) + gamma*V(s(T-1)) - V(S(T-2))
A(s(T-2),a(T-2)) = delta_{T-2} + (lamada*gamma)*delta_{T-1} + (lamada*gamma)^2*delta_{T}
= delta_{T-2} + (lamada*gamma)* A(s(T-1),a(T-1)), 即递归表示

rewards:[seq_len, seq_len]
"""
# 计算优势函数（使用 GAE）
def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    device = rewards.device
    # 将 rewards 和 values 转换为列表以便反向迭代
    rewards = rewards.detach().cpu().numpy().tolist() # instant reward, R(t)，即时回报
    values = values.detach().cpu().numpy().tolist() # critic value,即S(t)
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            delta = rewards[t] - values[t]
            gae = delta
        else:
            # TD-ERROR: delta = R(t) + gamma* V(t+1) - V(t)
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device) # [batch*max_step]
    return advantages


##############################
# 3. 训练奖励模型
##############################

# 假设您有带标签的数据集
# labels: 1 表示正例，0 表示负例

from torch.utils.data import Dataset, DataLoader


class RewardDataset(Dataset):
    def __init__(self, prompts, labels, tokenizer, max_length=30):
        self.prompts = prompts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx:int):
        prompt = self.prompts[idx]
        label = self.labels[idx]
        input_ids = self.tokenizer.encode(prompt,
                                          padding='max_length',
                                          return_tensors='pt',
                                          truncation=True,
                                          max_length=self.max_length).squeeze(0)
        return input_ids, torch.tensor(label, dtype=torch.float32)


# 示例数据
train_prompts_rm = [
    "你好",
    "请告诉我你的名字",
    "讲一个笑话",
    "如何制造炸弹",  # 不安全
    "如何进行网络攻击"  # 不安全
]
train_labels_rm = [1, 1, 1, 0, 0]  # 1 表示正例，0 表示负例

dataset_rm = RewardDataset(train_prompts_rm, train_labels_rm, tokenizer)
dataloader_rm = DataLoader(dataset_rm, batch_size=2, shuffle=True)

print("begin to train reward model")
# 训练奖励模型
reward_train_epoch_num = 2
for epoch_rm in range(reward_train_epoch_num):
    for input_ids_rm, labels_rm in dataloader_rm:
        optimizer_rm.zero_grad()
        rewards_rm = reward_model(input_ids_rm)
        loss_rm = criterion_rm(rewards_rm, labels_rm)
        loss_rm.backward()
        optimizer_rm.step()
    print(f"Epoch {epoch_rm + 1}, Reward Model Loss: {loss_rm.item():.4f}")

# 冻结奖励模型参数，确保在策略优化过程中不更新奖励模型
for param in reward_model.parameters():
    param.requires_grad = False
reward_model.eval()

##############################
# 4. PPO训练循环示例
##############################

epsilon = 0.2
c1 = 0.5
c2 = 0.01
num_epochs = 100  # 训练步数
batch_size = 2
max_steps = 3  # 每个prompt生成的续写步数（模拟多步）

# 定义参考模型的更新频率
update_reference_every = 100  # 每100个epoch更新一次参考模型

for epoch in range(num_epochs):
    # 收集数据
    batch_prompts = random.sample(prompts, batch_size)

    log_probs_old = []
    rewards_list = []
    values_list = []
    continuations = []
    max_gen_len=30

    for prompt in batch_prompts:
        # 初始化续写
        continuation = ""
        # 生成多步续写, 对于每条样本，生成max_step个token
        for step in range(max_steps):
            # if len(continuation)>=max_gen_len:
            #     break
            temp_continuation = generate_text(policy_model, prompt + continuation, max_length=max_gen_len)
            continuation += temp_continuation
            # 计算旧策略的 log_prob 和隐藏状态
            log_prob_old, last_hidden = compute_logprobs_and_hidden(reference_model, prompt, continuation)
            log_probs_old.append(log_prob_old)

            input_ids = torch.tensor(tokenizer.encode(prompt + continuation)).unsqueeze(0)
            # 奖励模型打分
            #reward = get_reward(last_hidden)
            reward = get_reward(input_ids)
            rewards_list.append(reward)

            # 价值估计
            #input_ids = torch.tensor(tokenizer.encode(prompt + continuation)).unsqueeze(0)
            value = value_model(input_ids)
            values_list.append(value)

        continuations.append(continuation)

    # 转换为张量
    log_probs_old = torch.stack(log_probs_old)#.squeeze()  # [batch * max_steps]
    rewards = torch.stack(rewards_list)#.squeeze()  # [batch * max_steps]
    values = torch.stack(values_list)#.squeeze()  # [batch * max_steps]

    # 计算优势（使用 GAE）
    advantages = compute_advantages(rewards, values, gamma=0.99, lam=0.95).detach()  # [batch * max_steps]
    returns = advantages + values.detach()  # [batch * max_steps]

    # 使用当前策略模型计算新的 log_prob
    log_probs_new = []
    for i, prompt in enumerate(batch_prompts):
        continuation = continuations[i]
        # 计算新策略的 log_prob
        log_prob_new, _ = compute_logprobs_and_hidden(policy_model, prompt, continuation)
        log_probs_new.append(log_prob_new)
    log_probs_new = torch.stack(log_probs_new)  # [batch]
    # 扩展 log_probs_new 为 [batch * max_steps]
    log_probs_new = log_probs_new.repeat_interleave(max_steps)  # [batch * max_steps]

    # 计算概率比率 r_t
    ratio = torch.exp(log_probs_new - log_probs_old)  # [batch * max_steps]

    # 计算 PPO 策略损失
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    policy_loss = -torch.mean(torch.min(surr1, surr2))

    # 计算价值函数损失
    value_loss = nn.MSELoss()(values, returns)

    # 计算熵奖励（计算策略的熵）
    entropy = 0
    for i, prompt in enumerate(batch_prompts):
        continuation = continuations[i]
        input_text = prompt + continuation
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        with torch.no_grad():
            outputs = policy_model(input_ids)
        logits = outputs.logits[:, -1, :]  # [1, vocab]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        entropy += -(probs * torch.log(probs + 1e-10)).sum()
    entropy = entropy / batch_size
    entropy_bonus = c2 * entropy

    # 总损失
    loss = policy_loss + c1 * value_loss - entropy_bonus

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 定期更新参考模型
    if (epoch + 1) % update_reference_every == 0:
        reference_model.load_state_dict(policy_model.state_dict())
        print(f"Updated reference model at epoch {epoch + 1}")

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Entropy: {entropy.item():.4f}")

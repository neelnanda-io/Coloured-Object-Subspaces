# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained("gpt2-large")

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab
# %%
COLORS = ["red", "purple", "blue", "black", "yellow", "brown", "green", "white"]
OBJECTS =  ["pencil", "notebook", "pen", "cup", "plate", "jug", "mug", "puzzle", "textbook", "leash",
"necklace", "bracelet", "bottle", "ball", "envelope", "lighter", "bowl"]
TEMPLATE = "Q: On the table, I see a {c1} {o1}, a {c2} {o2}, and a {c3} {o3}. What color is the {oq}?\nA:"
def make_single_prompt(append_ans=False):
    c1, c2, c3 = random.sample(COLORS, 3)
    o1, o2, o3 = random.sample(OBJECTS, 3)
    q_index = random.randint(0, 2)
    oq = [o1, o2, o3][q_index]
    cq = [c1, c2, c3][q_index]
    cq = cq
    prompt = TEMPLATE.format(c1=c1, c2=c2, c3=c3, o1=o1, o2=o2, o3=o3, oq=oq)
    if append_ans:
        return prompt+" "+cq
    else:
        return prompt, cq

print(make_single_prompt())
print(make_single_prompt(True))

def make_kshot_prompt(k=1):
    prompts = []
    final_prompt, final_ans = make_single_prompt(False)
    while len(prompts)<k:
        new_prompt = make_single_prompt(True)
        if new_prompt.split(" ")[-1] != final_ans:
            prompts.append(new_prompt)
    prompts.append(final_prompt)
    return "\n".join(prompts), final_ans
Q, A = (make_kshot_prompt(20))
print(Q)
print(A)
# %%
# colored_tokens = torch.tensor([model.to_single_token(" "+c) for c in COLORS]).cuda()
# records = []
# for k in tqdm.tqdm([1]):
#     for i in tqdm.trange(500):
#         Q, A = make_kshot_prompt(k)
#         A_tok = model.to_single_token(" "+A)

#         logits = model(Q)
#         log_probs = logits[0, -1].log_softmax(-1)

#         colored_tokens_excl_A = colored_tokens[colored_tokens!=A_tok]

#         record = {
#             "k": k,
#             "i": i,
#             "is_top": (log_probs.argmax() == A_tok).item(),
#             "is_top_color": (log_probs[colored_tokens_excl_A].max()<log_probs[A_tok]).item(),
#             "margin_top_color": (log_probs[A_tok] - log_probs[colored_tokens_excl_A].max()).item(),
#             "color": A
#         }
#         records.append(record)

# # %%
# df = pd.DataFrame(records)
# # display(df.groupby("k").mean())
# # px.box(df, x="k", y="margin_top_color")
# df["color"] = df["color"].astype("category")
# display(df.groupby("color")[["is_top_color", "margin_top_color", "is_top"]].mean())
# display(df.groupby("color")[["is_top_color", "margin_top_color", "is_top"]].count())

# %%
batch_size = 80
kshot = 1
random.seed(425342)
prompt_pairs = [make_kshot_prompt(kshot) for _ in range(batch_size)]
prompts = [p[0] for p in prompt_pairs]
answers = [" "+p[1] for p in prompt_pairs]
tokens = model.to_tokens(prompts)
answer_tokens = model.to_tokens(answers, prepend_bos=False).squeeze(-1)
print(model.to_str_tokens(tokens[0]))
print(model.to_str_tokens(answer_tokens[0]))
print(tokens.shape)
print(answer_tokens.shape)

_ = model.to_str_tokens(tokens[0])
for i in range(63):
    print(i, _[i])
FINAL = 62
OQ = 58
O1 = 43
O2 = 47
O3 = 52
C1 = 42
C2 = 46
C3 = 51
# %%
base_logits, base_cache = model.run_with_cache(tokens)

# %%
colored_tokens = torch.tensor([model.to_single_token(" "+c) for c in COLORS]).cuda()
W_U_color = model.W_U[:, colored_tokens]
print(W_U_color.shape)
W_U_ans = model.W_U[:, answer_tokens]
print(W_U_ans.shape)
W_U_ans_centered = (W_U_ans - W_U_color.mean(-1, keepdim=True))*8/7
print(W_U_ans_centered.shape)
# %%
PREV_ANS = 31
prev_ans_tokens = tokens[:, PREV_ANS]
W_U_ans_v_prev = W_U_ans - model.W_U[:, prev_ans_tokens]

incorrect_color_tokens = []
for b in range(batch_size):
    color_tokens_temp = tokens[b, [C1, C2, C3]]
    color_tokens_temp = color_tokens_temp[color_tokens_temp!=answer_tokens[b]]
    incorrect_color_tokens.append(color_tokens_temp)
incorrect_color_tokens = torch.stack(incorrect_color_tokens)
# print(model.to_str_tokens(incorrect_color_tokens[0]))
# print(model.to_string(tokens[0]))

W_U_ans_v_incorrect = W_U_ans - model.W_U[:, incorrect_color_tokens].mean(-1)

# %%
resid_stack, labels = base_cache.decompose_resid(-1, apply_ln=True, pos_slice=-1, return_labels=True)
print(resid_stack.shape)
dla_uncent = einops.einsum(resid_stack, W_U_ans, "comp batch d_model, d_model batch -> comp") / batch_size
dla_cent = einops.einsum(resid_stack, W_U_ans_centered, "comp batch d_model, d_model batch -> comp") / batch_size
dla_v_prev = einops.einsum(resid_stack, W_U_ans_v_prev, "comp batch d_model, d_model batch -> comp") / batch_size
dla_v_incorrect = einops.einsum(resid_stack, W_U_ans_v_incorrect, "comp batch d_model, d_model batch -> comp") / batch_size
line([dla_uncent, dla_cent, dla_v_prev, dla_v_incorrect], x=labels, title="DLA by Layer", line_labels=["Uncentered", "Centered", "prev", "incorrect"])
# %%
dla_labels = ["Uncentered", "Centered", "prev", "incorrect"]
all_z = base_cache.stack_activation("z")[:, :, -1, :, :]
W_U_ans_stack = torch.stack([W_U_ans, W_U_ans_centered, W_U_ans_v_prev, W_U_ans_v_incorrect], dim=0)
print(W_U_ans_stack.shape)
head_dla = einops.einsum(all_z, model.W_O, W_U_ans_stack, "layer batch head d_head, layer head d_head d_model, type d_model batch -> type layer head") / batch_size
imshow(head_dla, facet_col=0, facet_labels=dla_labels, title="DLA by Head")
line(head_dla.reshape(4, -1), x=model.all_head_labels(), title="DLA by Head", line_labels=dla_labels)

# %%
head_df = pd.DataFrame({
    "L": [l for l in range(n_layers) for h in range(n_heads)],
    "H": [h for l in range(n_layers) for h in range(n_heads)],
    "label": model.all_head_labels(),
})
for c, lab in enumerate(dla_labels):
    head_df[lab] = to_numpy(head_dla[c].reshape(-1))
nutils.show_df(head_df.sort_values("Centered", ascending=False).head(50))
# %%
layer = 23
head = 13
imshow(base_cache["pattern", layer][:, head, -1, :], x=nutils.process_tokens_index(prompts[0]))
# %%
all_patterns = base_cache.stack_activation("pattern")
print(all_patterns.shape)
head_df["bos_attn"] = to_numpy(einops.reduce(all_patterns[:, :, :, -1, 0], "layer batch head -> (layer head)", "mean"))
head_df["prev_attn"] = to_numpy(einops.reduce(all_patterns[:, :, :, -1, PREV_ANS], "layer batch head -> (layer head)", "mean"))
head_df["final_attn"] = to_numpy(einops.reduce(all_patterns[:, :, :, -1, 58:].sum(-1), "layer batch head -> (layer head)", "mean"))
# answer_pos = []
# for b in range(batch_size):
#     answer_pos.append(np.where(tokens[b]==answer_tokens[b]))

# head_df["corr_attn"] = to_numpy(einops.reduce(all_patterns[:, np.arange(batch_size), :, -1, answer_tokens], "batch layer head -> (layer head)", "mean"))
# head_df["corr_attn"] = to_numpy((einops.reduce(all_patterns[:, np.arange(batch_size), :, -1, ], "batch layer head -> (layer head)", "mean") + einops.reduce(all_patterns[:, np.arange(batch_size), :, -1, answer_tokens], "batch layer head -> (layer head)", "mean")))

# %%
corr_pos = []
incorr_pos = []
for b in range(batch_size):
    corr_pos.append(torch.arange(tokens.shape[-1]).cuda()[tokens[b]==answer_tokens[b]][-1].item())
    x = torch.tensor([C1, C2, C3]).cuda()
    incorr_pos.append(x[x!=corr_pos[-1]])
corr_pos = torch.tensor(corr_pos).cuda()
incorr_pos = torch.stack(incorr_pos)
corr_pos[0], incorr_pos[0]
# %%
head_df["corr_attn"] = to_numpy(einops.reduce(all_patterns[:, np.arange(batch_size), :, -1, corr_pos], "batch layer head -> (layer head)", "mean"))
head_df["incorr_attn"] = (to_numpy(einops.reduce(all_patterns[:, np.arange(batch_size), :, -1, incorr_pos[:, 0]], "batch layer head -> (layer head)", "mean"))+to_numpy(einops.reduce(all_patterns[:, np.arange(batch_size), :, -1, incorr_pos[:, 1]], "batch layer head -> (layer head)", "mean")))/2
head_df["diff_attn"] = head_df["corr_attn"] - head_df["incorr_attn"]
# %%
nutils.show_df(head_df.sort_values("incorrect", ascending=False).head(30))
# %%

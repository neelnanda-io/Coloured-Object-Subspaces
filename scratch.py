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
    final_prompt_cols = [c for c in final_prompt.split(" ") if c in COLORS]
    # print(final_prompt_cols)
    while len(prompts)<k:
        new_prompt = make_single_prompt(True)
        if new_prompt.split(" ")[-1] not in final_prompt_cols:
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
batch_size = 16
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
clean_tokens = tokens.clone()
corr_tokens = tokens.clone()
clean_answers = answer_tokens.clone()
corr_answers = answer_tokens.clone()
for b in range(batch_size):
    objects = clean_tokens[b, [O1, O2, O3]]
    colors = clean_tokens[b, [C1, C2, C3]]
    clean_obj = tokens[b, OQ]
    restr_objects = objects[objects!=clean_obj]
    corr_tokens[b, OQ] = restr_objects[random.randint(0, 1)]
    corr_answers[b] = colors[objects==corr_tokens[b, OQ]]

print(model.to_string(clean_tokens[0]))
print(model.to_string(clean_answers[0]))
print(model.to_string(corr_tokens[0]))
print(model.to_string(corr_answers[0]))

# %%
clean_logits = model(clean_tokens)
corr_logits = model(corr_tokens)

line([clean_logits[np.arange(batch_size), -1, clean_answers] - clean_logits[np.arange(batch_size), -1, corr_answers], corr_logits[np.arange(batch_size), -1, clean_answers] - corr_logits[np.arange(batch_size), -1, corr_answers]])

CLEAN_BASELINE_DIFF = (clean_logits[np.arange(batch_size), -1, clean_answers] - clean_logits[np.arange(batch_size), -1, corr_answers]).mean()
CORR_BASELINE_DIFF = (corr_logits[np.arange(batch_size), -1, clean_answers] - corr_logits[np.arange(batch_size), -1, corr_answers]).mean()
print("Clean Baseline Diff:", CLEAN_BASELINE_DIFF)
print("Corr Baseline Diff:", CORR_BASELINE_DIFF)

def metric(logits):
    logit_diff = (logits[np.arange(batch_size), -1, clean_answers] - logits[np.arange(batch_size), -1, corr_answers]).mean()
    return (logit_diff - CORR_BASELINE_DIFF) / (CLEAN_BASELINE_DIFF - CORR_BASELINE_DIFF)
# %%
filter_not_qkv_input = lambda name: "_input" not in name and "_result" not in name and "_attn_in" not in name and "_mlp_in" not in name
def get_cache_fwd_and_bwd(model, tokens, metric):
    model.reset_hooks()
    cache = {}
    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()
    model.add_hook(filter_not_qkv_input, forward_cache_hook, "fwd")

    grad_cache = {}
    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()
    model.add_hook(filter_not_qkv_input, backward_cache_hook, "bwd")

    torch.set_grad_enabled(True)
    value = metric(model(tokens))
    value.backward()
    model.reset_hooks()
    torch.set_grad_enabled(False)
    return value.item(), ActivationCache(cache, model), ActivationCache(grad_cache, model)

clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(model, clean_tokens, metric)
print("Clean Value:", clean_value)
print("Clean Activations Cached:", len(clean_cache))
print("Clean Gradients Cached:", len(clean_grad_cache))
corr_value, corr_cache, corr_grad_cache = get_cache_fwd_and_bwd(model, corr_tokens, metric)
print("corr Value:", corr_value)
print("corr Activations Cached:", len(corr_cache))
print("corr Gradients Cached:", len(corr_grad_cache))
# %%
denoise_cache = ActivationCache({k: corr_grad_cache[k] * (clean_cache[k] - corr_cache[k]) for k in clean_cache.cache_dict}, model)
noise_cache = ActivationCache({k: clean_grad_cache[k] * (clean_cache[k] - corr_cache[k]) for k in clean_cache.cache_dict}, model)
# %%
head_df["denoise"] = to_numpy(einops.reduce(denoise_cache.stack_activation("z"), "layer batch pos head d_head -> (layer head)", "sum") / batch_size)
head_df["denoise_final"] = to_numpy(einops.reduce(denoise_cache.stack_activation("z")[:, :, -1], "layer batch head d_head -> (layer head)", "sum") / batch_size)
head_df["noise"] = to_numpy(einops.reduce(noise_cache.stack_activation("z"), "layer batch pos head d_head -> (layer head)", "sum") / batch_size)
head_df["noise_final"] = to_numpy(einops.reduce(noise_cache.stack_activation("z")[:, :, -1], "layer batch head d_head -> (layer head)", "sum") / batch_size)

# %%
head_df["denoise_q"] = to_numpy(einops.reduce(denoise_cache.stack_activation("q"), "layer batch pos head d_head -> (layer head)", "sum") / batch_size)
head_df["denoise_k"] = to_numpy(einops.reduce(denoise_cache.stack_activation("k"), "layer batch pos head d_head -> (layer head)", "sum") / batch_size)
head_df["denoise_v"] = to_numpy(einops.reduce(denoise_cache.stack_activation("v"), "layer batch pos head d_head -> (layer head)", "sum") / batch_size)
head_df["denoise_pattern"] = to_numpy(einops.reduce(denoise_cache.stack_activation("pattern"), "layer batch head dest_pos src_pos -> (layer head)", "sum") / batch_size)
# %%
nutils.show_df(head_df.sort_values("denoise", ascending=False).head(30))
# %%
attn_pattern_attrib = clean_cache.stack_activation("pattern") * clean_grad_cache.stack_activation("pattern")

# %%
temp_df = head_df.sort_values("denoise", ascending=False).head(10)
for i in range(10):
    l = temp_df.iloc[i]["L"]
    h = temp_df.iloc[i]["H"]
    print(temp_df.iloc[i])
    imshow(attn_pattern_attrib[l, :, h, :, :].mean(0), x=nutils.process_tokens_index(prompts[0]), y=nutils.process_tokens_index(prompts[0]), title=f"Layer {l} Head {h}")
# %%
head_df["final_to_oq_attn"] = to_numpy(clean_cache.stack_activation("pattern")[:, :, :, FINAL, OQ].mean(1).flatten())
head_df["final_to_oq_attrib"] = to_numpy(attn_pattern_attrib[:, :, :, FINAL, OQ].mean(1).flatten())
nutils.show_df(head_df.sort_values("final_to_oq_attrib", ascending=False).head(10))
# %%
layers = [18, 18, 20]
heads = [13, 18, 3]

value_cache = [[], [], []]
def value_cache_hook(value, hook, index, pos=OQ):
    head = heads[index]
    value_cache[index].append(value[:, pos, head])
model.reset_hooks()
for i in range(3):
    model.blocks[layers[i]].attn.hook_v.add_hook(partial(value_cache_hook, index=i))

# model(tokens)
# model(tokens)
# model(tokens)

print(len(value_cache[0]))
# %%
prompts = []
answers = []
records = []
for i in range(3008):
    prompt, answer = make_kshot_prompt(1)
    prompts.append(prompt)
    answers.append(answer)
    words = prompt.split(" ")
    colors = [words[28], words[31], words[35]]
    objects = [words[29][:-1], words[32][:-1], words[36][:-1]]
    index = colors.index(answer)
    curr_object = objects[index]


    record = {
        "i": i,
        "color": answer,
        "object": curr_object,
        "pos": index,
        "prompt": prompt
    }
    records.append(record)
prompt_df = pd.DataFrame(records)
all_tokens = model.to_tokens(prompts)
for i in tqdm.trange(0, 3008, 64):
    tokens = all_tokens[i:i+64]
    model(tokens)
cached_values = torch.stack([torch.concat(value_cache[i]) for i in range(3)])
print(cached_values.shape)

# %%
prompt_df["color"] = prompt_df["color"].astype("category")
prompt_df["object"] = prompt_df["object"].astype("category")
# %%
cached_values_centered = cached_values - cached_values.mean(-2, keepdim=True)
# %%
prompt_df["object_index"] = prompt_df["object"].apply(lambda x: OBJECTS.index(x))
prompt_df["color_index"] = prompt_df["color"].apply(lambda x: COLORS.index(x))
prompt_df
# %%
cached_values_centered_train = cached_values_centered[:, prompt_df["i"]<2000]
cached_values_centered_test = cached_values_centered[:, prompt_df["i"]>=2000]
prompt_df_train = prompt_df[prompt_df["i"]<2000]
prompt_df_test = prompt_df[prompt_df["i"]>=2000]
# %%
value_means_by_color = torch.zeros((3, len(COLORS), 64)).cuda()
value_means_by_object = torch.zeros((3, len(OBJECTS), 64)).cuda()
value_means_by_pos = torch.zeros((3, 3, 64)).cuda()
for i in range((value_means_by_pos.shape[1])):
    value_means_by_pos[:, i, :] = cached_values_centered_train[:, prompt_df_train["pos"]==i].mean(1)
for i in range((value_means_by_color.shape[1])):
    value_means_by_color[:, i, :] = cached_values_centered_train[:, prompt_df_train["color_index"]==i].mean(1)
for i in range((value_means_by_object.shape[1])):
    value_means_by_object[:, i, :] = cached_values_centered_train[:, prompt_df_train["object_index"]==i].mean(1)
# %%
cached_values_centered_by_pos = cached_values_centered_test - value_means_by_pos[:, prompt_df_test["pos"].values, :]
px.box(to_numpy((cached_values_centered_by_pos.norm(dim=-1)/cached_values_centered_test.norm(dim=-1))).T).show()
cached_values_centered_by_color = cached_values_centered_test - value_means_by_color[:, prompt_df_test["color_index"].values, :]
px.box(to_numpy((cached_values_centered_by_color.norm(dim=-1)/cached_values_centered_test.norm(dim=-1))).T).show()
cached_values_centered_by_object = cached_values_centered_test - value_means_by_object[:, prompt_df_test["object_index"].values, :]
px.box(to_numpy((cached_values_centered_by_object.norm(dim=-1)/cached_values_centered_test.norm(dim=-1))).T).show()

records = []
for i in range(3):
    label = f"L{layers[i]}H{heads[i]}"
    for j in range(2000, 3008):
        for mode, values in [("pos", cached_values_centered_by_pos), ("color", cached_values_centered_by_color), ("object", cached_values_centered_by_object)]:
            records.append({
                "i": j,
                "mode": mode,
                "value": values[i, j-2000].norm().item(),
                "value_normed": (values[i, j-2000].norm() / cached_values_centered_test[i, j-2000].norm()).item(),
                "label": label
            })
df = pd.DataFrame(records)
df

# %%
px.box(df, x="label", color="mode", y="value_normed").show()
px.box(df, x="label", color="mode", y="value").show()
# %%
import sklearn.linear_model

probe = sklearn.linear_model.LogisticRegression(max_iter=1000)
probe
# %%
for index in range(3):
    for labels in ["pos", "color_index", "object_index"]:
        X = cached_values_centered_train[index]
        y = prompt_df_train[labels].values
        probe.fit(to_numpy(X), y)
        print(index, labels)
        print((probe.predict(to_numpy(cached_values_centered_test[index]))==prompt_df_test[labels].values).astype("float").mean())
# %%

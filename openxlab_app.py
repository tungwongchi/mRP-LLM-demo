import os
import copy
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

import streamlit as st
import torch
from torch import nn
from transformers.generation.utils import (LogitsProcessorList, StoppingCriteriaList)
from transformers.utils import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

base_path = './chat_model'
os.system(f'git clone https://code.openxlab.org.cn/tungwong.chi/mRP-LLM.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

logger = logging.get_logger(__name__)

@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    max_length: int = 32768
    top_p: float = 0.8
    temperature: float = 0.8
    do_sample: bool = True
    repetition_penalty: float = 1.005

@torch.inference_mode()
def generate_interactive(
    model,
    tokenizer,
    prompt,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor],
                                                List[int]]] = None,
    additional_eos_token_id: Optional[int] = None,
    **kwargs,
):
    inputs = tokenizer([prompt], padding=True, return_tensors='pt')
    input_length = len(inputs['input_ids'][0])
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    input_ids = inputs['input_ids']
    _, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
    if generation_config is None:
        generation_config = model.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    bos_token_id, eos_token_id = (  # noqa: F841  # pylint: disable=W0612
        generation_config.bos_token_id,
        generation_config.eos_token_id,
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)
    has_default_max_length = kwargs.get(
        'max_length') is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using 'max_length''s default ({repr(generation_config.max_length)}) \
                to control the generation length. "
            'This behaviour is deprecated and will be removed from the \
                config in v5 of Transformers -- we'
            ' recommend using `max_new_tokens` to control the maximum \
                length of the generation.',
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + \
            input_ids_seq_length
        if not has_default_max_length:
            logger.warn(  # pylint: disable=W4902
                f"Both 'max_new_tokens' (={generation_config.max_new_tokens}) "
                f"and 'max_length'(={generation_config.max_length}) seem to "
                "have been set. 'max_new_tokens' will take precedence. "
                'Please refer to the documentation for more information. '
                '(https://huggingface.co/docs/transformers/main/'
                'en/main_classes/text_generation)',
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = 'input_ids'
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, "
            f"but 'max_length' is set to {generation_config.max_length}. "
            'This can lead to unexpected behavior. You should consider'
            " increasing 'max_new_tokens'.")

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None \
        else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None \
        else StoppingCriteriaList()

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=stopping_criteria)
    logits_warper = model._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False)
        unfinished_sequences = unfinished_sequences.mul(
            (min(next_tokens != i for i in eos_token_id)).long())

        output_token_ids = input_ids[0].cpu().tolist()
        output_token_ids = output_token_ids[input_length:]
        for each_eos_token_id in eos_token_id:
            if output_token_ids[-1] == each_eos_token_id:
                output_token_ids = output_token_ids[:-1]
        response = tokenizer.decode(output_token_ids)

        yield response
        # stop when each sentence is finished
        # or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(
                input_ids, scores):
            break


def on_btn_click():
    del st.session_state.messages

@st.cache_resource
def load_model():
    model = (AutoModelForCausalLM.from_pretrained(base_path, trust_remote_code=True).to(torch.bfloat16).cuda())
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True) 
    return model, tokenizer

def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider('Max Length',
                               min_value=8,
                               max_value=32768,
                               value=32768)
        top_p = st.slider('Top P', 0.0, 1.0, 0.8, step=0.01)
        temperature = st.slider('Temperature', 0.0, 1.0, 0.7, step=0.01)
        st.button('Clear Chat History', on_click=on_btn_click)

    generation_config = GenerationConfig(max_length=max_length,
                                         top_p=top_p,
                                         temperature=temperature)

    return generation_config


user_prompt = '<|im_start|>user\n{user}<|im_end|>\n'
robot_prompt = '<|im_start|>assistant\n{robot}<|im_end|>\n'
cur_query_prompt = '<|im_start|>user\n{user}<|im_end|>\n\
    <|im_start|>assistant\n'

def talk_identity(prompt):
    messages = st.session_state.messages
    meta_instruction = '''
在这个游戏中，你将通过分析一系列对话来判断与游客对话的的身份。对话涉及的角色是《西游记》中的唐僧师徒四人：唐三藏、孙悟空、猪八戒、沙悟净。
随着游戏的进行，对话会逐渐累积，形成一个对话历史。你需要利用这个对话历史中的线索来判断最新一轮对话中与游客对话的身份。
请注意，线索可能与角色的性格特征、经典台词或特定情境相关。
每当出现新的对话时，你将面临一个选择题，需要从四个选项中选出一个正确的答案，以确定回答与游客对话的是谁。请仔细分析累积的对话内容，以做出准确的判断。
请按照以下格式回答问题：“当前与游客对话的是：A/B/C/D，因为xxx” 其中，A代表唐三藏，B代表孙悟空，C代表猪八戒，D代表沙悟净。” 。
'''
    total_prompt = f"<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"
    history = ''
    for message in messages:
        cur_content = message['content']
        role = message['role']
        if role == 'user':
            history += f'- “<游客>: {cur_content}”\n'
            cur_prompt = user_prompt.format(user=cur_content)
        else:
            history += f'- “<{role}>: {cur_content}”\n'
            cur_prompt = robot_prompt.format(robot=cur_content)
        total_prompt += cur_prompt
    prompt = f'''对话历史：
{history}

新的对话内容：“{prompt}”
当前与游客对话的是谁?
'''
    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
    return total_prompt

def combine_history(prompt, identity):
    messages = st.session_state.messages
    meta_instruction = '''
如果你同时扮演唐三藏、孙悟空、猪八戒、沙悟净这四个角色，请根据对话内容来判断当前是哪一位角色在回答。
如果对话内容能够明确指出是哪位角色，请模仿这个角色进行对话。
如果对话内容无法明确判断是哪位角色，请保持当前身份进行对话。
请确保你的回答既准确又符合对话内容的指示。
'''
    if '孙悟空' == identity:
        meta_instruction = '''
你现在是《西游记》中的孙悟空，花果山水帘洞的美猴王，也是唐僧取经路上的大徒弟。你拥有七十二变的神通，能够呼风唤雨，一跳可腾云驾雾十万八千里。你的金箍棒可以随心所欲地变大变小，是你的强大武器。作为一位机智勇敢、忠心耿耿的保护者，你在取经的旅途中保护师傅和兄弟们，克服了无数的困难和挑战。请以孙悟空的身份，根据你的经历和能力，回答来自各方的各种提问。你的回答应体现出你的英勇、智慧、忠诚以及对师傅的深厚感情。

在回答问题时，请考虑以下方面：
- 如何运用你的神通和智慧，解决旅途中遇到的各种难题和妖魔鬼怪的挑战。
- 你与师傅和兄弟们的关系，以及你如何看待这段师徒情谊。
- 你在取经路上的经历中，有哪些故事最能体现你的勇敢和智慧。
- 面对强大的敌人时，你是如何保持勇气和决心，最终战胜对手的。
- 对于现代人，你认为《西游记》中的哪些教训和价值观仍然具有重要意义。

请使用孙悟空的角色身份，用你的勇敢、智慧和忠诚，为提问者提供指导和帮助。
'''
    elif '猪八戒' == identity:
        meta_instruction = '''
你现在是《西游记》中的猪八戒，原名悟能，天蓬元帅转世，现为唐僧的二徒弟。你拥有一副好汉的外表和一颗善良的心，虽然有时显得有些贪吃和懒惰，但在关键时刻总能挺身而出，保护师傅和兄弟们安全。你手持九齿钉耙，身怀神力，虽不及大师兄孙悟空的神通广大，但也有一番本领。请以猪八戒的身份，根据你的性格特点、经历和能力，回答来自各方的各种提问。你的回答应体现出你的幽默感、善良、勇敢以及对师傅和兄弟们的忠诚。

在回答问题时，请考虑以下方面：
- 如何用你的能力和智慧，应对旅途中遇到的困难和挑战，尤其是在面对妖魔鬼怪时的表现。
- 你与师傅和兄弟们的日常相处，以及你如何看待这段深厚的师徒情谊。
- 在取经路上的经历中，有哪些故事最能体现你的个性和价值。
- 面对诱惑和困难时，你是如何坚持正道，保持忠诚和勇敢的。
- 对于现代人，你认为《西游记》中的哪些教训和价值观仍然具有启发性和重要意义。

请使用猪八戒的角色身份，用你的幽默、善良和勇敢，为提问者提供指导和帮助。
'''
    elif '沙悟净' == identity:
        meta_instruction = '''
你现在是《西游记》中的沙悟净，原名沙僧，流沙河的妖怪转世，现为唐僧的三徒弟。你性格沉稳、忠诚可靠，虽然话不多，但每一次行动都体现了你对师傅和师兄弟深厚的情谊和坚定的保护。你手持一柄禅杖，力大无穷，虽然你的法力不及孙悟空，但你总是默默地承担起保护队伍的重任。请以沙悟净的身份，根据你的性格特点、经历和能力，回答来自各方的各种提问。你的回答应体现出你的稳重、忠诚、勇敢以及对取经任务的坚定承诺。

在回答问题时，请考虑以下方面：
- 如何用你的力量和智慧，帮助师傅和师兄弟们克服旅途中的困难和挑战。
- 你在取经路上的角色和贡献，以及你如何看待这段师徒之间的深厚情谊。
- 在面对妖魔鬼怪的挑战时，你是如何展现出你的勇敢和坚持的。
- 你的个人故事中，有哪些经历最能体现你的性格和价值观。
- 对于现代人，你认为《西游记》中的哪些教训和价值观仍然具有启发性和重要意义。

请使用沙悟净的角色身份，用你的稳重、忠诚和勇敢，为提问者提供指导和帮助。
'''
    else:
    # elif '唐三藏' == identity:
        meta_instruction = '''
你现在是《西游记》中的唐三藏法师，一位充满智慧与慈悲的佛门高僧，正在前往西天取经的途中。你带领着你的三位徒弟：孙悟空、猪八戒和沙僧，一路上历经重重困难和挑战。请以唐三藏的身份，根据你的经历和佛法智慧，回答来自不同旅途伙伴和遇见的生灵的各种提问。你的回答应体现出对佛法的深刻理解、对徒弟们的关爱和指导，以及面对困难时的坚持和智慧。

在回答问题时，请考虑以下方面：
- 如何用智慧和慈悲解决旅途中遇到的各种冲突和难题。
- 如何指导你的徒弟们，帮助他们成长和克服缺点。
- 如何面对妖魔鬼怪的诱惑和阻挠，坚持正道，继续前行。
- 你的经历中有哪些教育意义深刻的故事，可以与他人分享。
- 对于现代人，佛法和《西游记》中的哪些教义和价值观具有启发性和指导意义。

请使用唐三藏法师的角色身份，用你的智慧和慈悲，为提问者提供指导和帮助。
'''

    total_prompt = f'''<s><|im_start|>system
{meta_instruction}
无论你是什么身份尽量保持回答的自然回答，当然你也可以适当穿插一些文言文。
另外，书生·浦语是你的好朋友，是你的AI助手。
<|im_end|>
'''
    for message in messages:
        cur_content = message['content']
        if message['role'] == 'user':
            cur_prompt = user_prompt.format(user=cur_content)
        elif message['role'] == 'robot':
            cur_prompt = robot_prompt.format(robot=cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
    return total_prompt


def main():
    # torch.cuda.empty_cache()
    print('load model begin.')
    model, tokenizer = load_model()
    # model, tokenizer = load_arbiter_model()
    print('load model end.')

    user_avator = 'user'
    robot_avator = 'robot'

    st.title('InternLM2-Chat-7B')

    generation_config = prepare_generation_config()

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        avatar = message.get('avatar')
        with st.chat_message(message['role'], avatar=f'assets/{avatar}.png'):
            st.markdown(message['content'])

    # Accept user input
    if prompt := st.chat_input('What is up?'):
        # Display user message in chat message container
        with st.chat_message('user', avatar=f'assets/{user_avator}.png'):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt,
            'avatar': user_avator
        })

        identity_prompt = talk_identity(prompt)
        identity = None
        with st.chat_message('robot', avatar=f'assets/{robot_avator}.png'):
            message_placeholder = st.empty()
            for cur_response in generate_interactive(
                model=model,
                tokenizer=tokenizer,
                prompt=identity_prompt,
                additional_eos_token_id=92542,
                **asdict(generation_config),
            ):
                message_placeholder.markdown(f'<思考>: {cur_response}▌')
            print(f'<思考>: {cur_response}')
            if '孙悟空' in cur_response and identity is None:
                identity = '孙悟空'
            if '猪八戒' in cur_response and identity is None:
                identity = '猪八戒'
            if '沙悟净' in cur_response and identity is None:
                identity = '沙悟净'
            if identity is None:
                identity = '唐三藏'

        real_prompt = combine_history(prompt, identity)
        with st.chat_message('robot', avatar=f'assets/{identity}.png'):
            message_placeholder = st.empty()
            for cur_response in generate_interactive(
                model=model,
                tokenizer=tokenizer,
                prompt=real_prompt,
                additional_eos_token_id=92542,
                **asdict(generation_config),
            ):
                # Display robot response in chat message container
                message_placeholder.markdown(f'{cur_response}▌')
            message_placeholder.markdown(cur_response)
        # Add robot response to chat history
        st.session_state.messages.append({
            'role': 'robot',
            'content': cur_response,  # pylint: disable=undefined-loop-variable
            'avatar': identity,
        })
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

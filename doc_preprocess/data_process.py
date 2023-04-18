import re

def remove_html_tag(html_string):
    """
    remove html tag
    :param html_string:
    :return:
    """
    return re.sub('<.*?>', '', html_string)

def convert_html2FAQ(html_string):
    """
    convert html to FAQ
    :param html_string:
    :return: question and answer tuple list

    input example : 满足什么条件才能进行全息幻视?	<div dir="ltr" data-zone-id="0" data-line-index="0" data-line="true">您好，全息幻视需要满足以下两个基本条件：</div> <div data-zone-id="0" data-line-index="1" data-line="true"> </div> <div data-zone-id="0" data-line-index="2" data-line="true">1)您已经拥有了想要幻视的目标城市外观，且当前城市外观和目标城市外观都需要是永久城市外观；</div> <div data-zone-id="0" data-line-index="3" data-line="true">2)目标城市外观假如为可升级类型的城堡皮肤，需要达到最高等级后，才能够被幻视。</div>
    """
    lines = html_string.split('\n')
    lines = [line.strip() for line in lines if line.strip() != '']
    for line in lines:
        tuple = line.split('\t', 2)
        question, answer = tuple[0], tuple[1]
        answer = remove_html_tag(answer)
        yield f"Question: {question}", f"Answer: {answer}"

with open('JY_FAQ.txt', 'w', encoding='utf-8') as out_f:
    with open('JY_FAQ_Origin.txt', 'r', encoding='utf-8') as input_f:
        all_html = input_f.read()
        for question, answer in convert_html2FAQ(all_html):
            out_f.write(f"\n{question}\n{answer}\n")
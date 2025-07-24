from intellitube.agents.intellitube_agent.agent import test_agent, chat_loop
# test_agent()
chat_loop()
exit(0)



from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from intellitube.llm import init_llm
from intellitube.agents.intellitube_agent.prompts import (
    multi_query_prompt
)
from intellitube.agents.intellitube_agent.states import (
    DocumentData, MultiQueryData, RetrieverNodeState,
    QueryExtractorResponseState
)

llm = init_llm('google')

def multiquery_gen_node(state: RetrieverNodeState) -> RetrieverNodeState:
    messages = ChatPromptTemplate.from_messages(
        [multi_query_prompt, state.query]
    )
    structured_llm = llm.with_structured_output(MultiQueryData)
    query_data = structured_llm.invoke(messages.format_messages(summary=state.data["summary"]))
    state.query_data = query_data
    return state


if __name__ == '__main__':
    mq = multiquery_gen_node(
        RetrieverNodeState(
            query="What is the economic impacts of climate change?",
            data=DocumentData(
                metadata=QueryExtractorResponseState(
                    instruction="", analysis="", url="", urlof="document"
                ),
                documents=[
                    Document("The speaker begins by highlighting the economic impacts of climate change, mentioning how rising temperatures and sea levels could disrupt agriculture and coastal communities. He then transitions into a discussion about government policies, critiquing current efforts as insufficient and driven more by political theater than real change."),
                    Document('The speaker praises a few nations that have taken meaningful steps, such as subsidizing renewable energy and enforcing carbon taxes. However, he also warns that without global cooperation, these efforts might not be enough. He proposes a multi-tiered plan involving both national legislation and international treaties.'),
                    Document('In the second half of the video, he debunks common myths around climate change, especially the idea that it\'s "too late to act." He presents data from recent IPCC reports showing that aggressive mitigation efforts could still significantly reduce harm.'),
                    Document('The video ends with a motivational call to action aimed at younger generations, urging them to stay informed, vote consciously, and push for systemic reform rather than isolated lifestyle changes.)'),
                ],
                summary="The video discusses the economic risks of climate change, criticizes weak government policies, highlights successful initiatives in some countries, and proposes a collaborative global solution. It also debunks defeatist climate myths and ends with a call to action for systemic reform, especially targeting youth."
            )
        )
    )

    from pprint import pprint
    pprint(mq.model_dump())
    print("\n\n\n" + "==" * 40 + "\n\n\n")
    print(mq.query_data)


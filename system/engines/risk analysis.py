import streamlit as st
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config


# 定义绘图函数
# tci结果绘图
def plot_tci(csv_file):
    st.write(
        "The total connectedness index (TCI) illustrates the average impact a shock in one series has on all others.")
    # Load data from CSV file
    data = pd.read_csv(csv_file)

    # Convert date column to datetime type
    data['Date'] = pd.to_datetime(data['Date'])

    # Set date column as index
    data.set_index('Date', inplace=True)

    # Plot time series chart in Streamlit
    st.title('The total connectedness index')
    st.line_chart(data['TCI'])


# net结果绘图
def plot_net(csv_file):
    st.write(
        "the net total directional connectedness(NET), NETi, which illustrates the net influence on the predefined network. If NETi>0(NETi<0), we know that the impact series i has on all others is larger (smaller) than the impact all others have on series i. Thus, series i is considered as a net transmitter (receiver) of shocks and hence driving (driven by) the network.")
    # 从CSV文件加载数据
    data = pd.read_csv(csv_file)

    # 将日期列转换为日期时间类型
    data['Date'] = pd.to_datetime(data['Date'])

    # 设置日期列作为索引
    data.set_index('Date', inplace=True)

    # 获取数据的列名
    columns = data.columns

    # 在 Streamlit 中绘制时间序列图
    st.title('The net total directional connectedness')

    # 将图表分为两列
    col_1, col_2 = st.columns(2)

    # 遍历每列数据并绘制时间序列图
    for i, column in enumerate(columns):
        # 根据索引的奇偶性决定图表的放置位置
        if i % 2 == 0:
            chart_col = col_1
        else:
            chart_col = col_2

        # 在当前列中绘制时间序列图
        chart_col.subheader("The net total directional connectedness of " + column)
        chart_col.line_chart(data[column], use_container_width=True)


# npdc结果绘图
def plot_npdc(csv_file):
    st.write(
        "On the bilateral level, the net pairwise directional connectedness measure(NPDC), NPSOij, is of major interest. If NPSOij>0 (NPSOij<0) it means that series i has a larger (smaller) impact on series j than series j has on series i.")

    # 从CSV文件加载数据
    data = pd.read_csv(csv_file)

    # 将日期列转换为日期时间类型
    data['Date'] = pd.to_datetime(data['Date'])

    # 设置日期列作为索引
    data.set_index('Date', inplace=True)

    # 获取数据的列名
    columns = data.columns

    # 在 Streamlit 中绘制时间序列图
    st.title('The net pairwise directional connectedness')

    # 将图表分为两列
    col_1, col_2 = st.columns(2)

    # 遍历每列数据并绘制时间序列图
    for i, column in enumerate(columns):
        # 根据索引的奇偶性决定图表的放置位置
        if i % 2 == 0:
            chart_col = col_1
        else:
            chart_col = col_2

        # 在当前列中绘制时间序列图
        chart_col.subheader("The net pairwise directional connectedness of " + column)
        chart_col.line_chart(data[column], use_container_width=True)


# pci网络结果绘图
def network_pci(nodes_data, edges_data):
    st.write(
        "PCI (Pairwise Connectedness Index) is an indicator used to measure the overall degree of connectedness between variables. It reflects the strength of the overall direct and indirect influence of a variable on other variables.The value of PCI ranges from [0, 100], with higher values indicating stronger connectivity between variables.")
    st.write("In an undirected network graph, the thickness of the connecting edges indicates the magnitude of the risk.")
    st.title('network plots which represents the PCI')
    # 将节点和边添加到列表中
    nodes = [Node(id=node, label=node, size=10, color='red') for node in nodes_data]

    # 获取权重的最大值和最小值，用于归一化
    min_strength = min([edge[2] for edge in edges_data])
    max_strength = max([edge[2] for edge in edges_data])

    # 添加边，并归一化权重
    edges = [
        Edge(
            source=edge[0],
            target=edge[1],
            label=f"{edge[2]:.2f}",
            width=10 * ((edge[2] - min_strength) / (max_strength - min_strength)),  # 归一化并放大权重以便在图中清晰可见
            length=300
        )
        for edge in edges_data
    ]

    # 创建图的配置
    config = Config(width=1000, height=800,
                    directed=False, nodeHighlightBehavior=True,
                    highlightColor="#F7A7A6", collapsible=True,
                    node={'labelProperty': 'label', 'color': 'lightblue'},
                    link={'highlightColor': '#F7A7A6', 'renderLabel': True},
                    label_font_size=18,
                    staticGraph=True)  # 添加 staticGraph=True 使得图形固定，不再有拖动或缩放的交互效果

    # 绘制图
    agraph(nodes=nodes, edges=edges, config=config)


# npdc网络结果绘图
def network_npdc(nodes, colors, sizes, edges):
    st.write(
        "NPDC (Net Pairwise Directional Connectedness) and PCI (Pairwise Connectedness Index) are indicators used to measure the degree of connectivity between variables in multivariate time series data.")
    st.write("In a directed network graph, yellow nodes are used to denote risk spillover countries, blue nodes are used to denote risk spillover countries, and the thickness of the connecting edges denotes the magnitude of the risk.")
    st.title('network plots which represents the NPDC')
    # Create node objects
    nodes_data = [
        Node(id=node, label=node, size=sizes[i], color=colors[i])
        for i, node in enumerate(nodes)
    ]

    # Create edge objects
    edges_data = [
        Edge(
            source=edge[0],
            target=edge[1],
            label=f"{edge[2]['weight']:.2f}",
            width=edge[2]['weight'],
            length=300,
        )
        for edge in edges
    ]

    # Create graph configuration
    config = Config(
        width=1000,
        height=800,
        directed=True,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=True,
        node={'labelProperty': 'label', 'color': 'lightblue'},
        link={'highlightColor': '#F7A7A6', 'renderLabel': True},
        label_font_size=18,
        staticGraph=True,
    )

    # Draw the graph
    agraph(nodes=nodes_data, edges=edges_data, config=config)


# pci网络结果绘图的数据
nodes_data_1 = ['Canada', 'France', 'Germany', 'Italy', 'Japan', 'Britain', 'America', 'WTI']
edges_data_1 = [
    ('Canada', 'France', 3.33328),
    ('Canada', 'Germany', 2.700347),
    ('Canada', 'Italy', 3.305393),
    ('Canada', 'Japan', 3.032046),
    ('Canada', 'Britain', 3.395712),
    ('Canada', 'America', 3.421573),
    ('Canada', 'WTI', 3.944357),
    ('France', 'Germany', 2.761257),
    ('France', 'Italy', 5),
    ('France', 'Japan', 3.083767),
    ('France', 'Britain', 3.407209),
    ('France', 'America', 3.262152),
    ('France', 'WTI', 3.668187),
    ('Germany', 'Italy', 2.828828),
    ('Germany', 'Japan', 2.681661),
    ('Germany', 'Britain', 2.617453),
    ('Germany', 'America', 2.802221),
    ('Germany', 'WTI', 3.051003),
    ('Italy', 'Japan', 3.196435),
    ('Italy', 'Britain', 3.546247),
    ('Italy', 'America', 2.854276),
    ('Italy', 'WTI', 3.641741),
    ('Japan', 'Britain', 3.216584),
    ('Japan', 'America', 2.951695),
    ('Japan', 'WTI', 3.581411),
    ('Britain', 'America', 3.166125),
    ('Britain', 'WTI', 3.776988),
    ('America', 'WTI', 3.279849)
]

nodes_data_2 = ['America', 'France', 'Britain', 'Italy', 'China', 'Germany']
edges_data_2 = [
    ('France', 'Britain', 3.409324),
    ('America', 'Italy', 1.450545),
    ('France', 'Italy', 5),
    ('Britain', 'Italy', 2.761331),
    ('America', 'Germany', 1.253296),
    ('France', 'Germany', 4.334928),
    ('Britain', 'Germany', 2.516086),
    ('Italy', 'Germany', 4.352448),
]

nodes_data_3 = ['stock', 'gas', 'oil', 'coal']
edges_data_3 = [
    ('stock', 'gas', 2.560771),
    ('stock', 'oil', 2.247668),
    ('stock', 'coal', 5),
    ('gas', 'oil', 2.799253),
    ('gas', 'coal', 2.357689),
    ('oil', 'coal', 2.994936),
]

# npdc网络结果绘图的数据
nodes_1 = ['Canada', 'France', 'Germany', 'Italy', 'Japan', 'Britain', 'America', 'WTI']
colors_1 = ['lightblue', 'yellow', 'lightblue', 'lightblue', 'yellow', 'lightblue', 'yellow', 'yellow']
sizes_1 = [0.653801081, 0.468311027, 0.003233689, 0.490388241, 0.080715935, 0.528448184, 0.126844233, 1.000000000]
sizes_1 = [num * 40 for num in sizes_1]
edges_1 = [
    ('Canada', 'America', {'weight': 3.419004}),
    ('Canada', 'WTI', {'weight': 4.232791}),
    ('Germany', 'WTI', {'weight': 1.368972}),
    ('Italy', 'WTI', {'weight': 1.513782}),
    ('Italy', 'France', {'weight': 5}),
    ('Japan', 'WTI', {'weight': 2.080485}),
    ('Britain', 'Japan', {'weight': 1.644266}),
    ('Britain', 'America', {'weight': 1.434383}),
    ('Britain', 'WTI', {'weight': 3.736664}),
    ('America', 'France', {'weight': 1.641651}),
]

nodes_2 = ['America', 'France', 'Britain', 'Italy', 'China', 'Germany']
colors_2 = ['yellow', 'lightblue', 'lightblue', 'yellow', 'lightblue', 'lightblue']
sizes_2 = [0.72412132, 1.00000000, 0.18383148, 0.75227564, 0.04680613, 0.24575935]
sizes_2 = [num * 40 for num in sizes_2]
edges_2 = [
    ('France', 'America', {'weight': 1.323429}),
    ('France', 'Germany', {'weight': 1.340446}),
    ('France', 'Italy', {'weight': 5}),
    ('Britain', 'America', {'weight': 1.512627}),
    ('Germany', 'Britain', {'weight': 2.319864}),
    ('Britain', 'France', {'weight': 1.655928}),
]

nodes_3 = ['stock', 'gas', 'oil', 'coal']
colors_3 = ['yellow', 'yellow', 'lightblue', 'lightblue']
sizes_3 = [1.0000000, 0.1896144, 0.4128260, 0.7767884]
sizes_3 = [num * 40 for num in sizes_3]
edges_3 = [
    ('coal', 'stock', {'weight': 5}),
]

# streamlit框架
col1, col2, col3 = st.columns((0.1, 4, 1))
with col2:
    st.markdown("# Risk Transmission Analysis")
    analysis_option = st.sidebar.radio('Analytical Perspectives', ['Global', 'National'])

if analysis_option == 'Global':
    energies = ['Oil', "Gas"]
    energy_option = st.selectbox('Energy Options', energies)

    if energy_option == 'Oil':
        figures = ['The Total Connectedness Index', "The Net Total Directional Connectedness",
                   "The Net Pairwise Directional Connectedness", "Network Plots which Represents the PCI",
                   "Network Plots which Represents the NPDC"]
        figure_option = st.selectbox('Please select a pattern', figures)
        st.markdown("<center><h1>G7_WTI system</h1></center>", unsafe_allow_html=True)

        if figure_option == "The Total Connectedness Index":
            plot_tci('../dataset/tci_1.csv')

        elif figure_option == "The Net Total Directional Connectedness":
            plot_net('../dataset/net_1.csv')



        elif figure_option == "The Net Pairwise Directional Connectedness":
            plot_npdc('../dataset/npdc_1.csv')



        elif figure_option == "Network Plots which Represents the PCI":
            network_pci(nodes_data_1, edges_data_1)



        elif figure_option == "Network Plots which Represents the NPDC":
            network_npdc(nodes_1, colors_1, sizes_1, edges_1)

    if energy_option == 'Gas':
        figures = ['The Total Connectedness Index', "The Net Total Directional Connectedness",
                   "The Net Pairwise Directional Connectedness", "Network Plots which Represents the PCI",
                   "Network Plots which Represents the NPDC"]
        figure_option = st.selectbox('Please select a pattern', figures)
        st.markdown("<center><h1>Gas system</h1></center>", unsafe_allow_html=True)
        if figure_option == "The Total Connectedness Index":
            plot_tci('../dataset/tci_2.csv')


        elif figure_option == "The Net Total Directional Connectedness":
            plot_net('../dataset/net_2.csv')



        elif figure_option == "The Net Pairwise Directional Connectedness":
            plot_npdc('../dataset/npdc_2.csv')



        elif figure_option == "Network Plots which Represents the PCI":
            network_pci(nodes_data_2, edges_data_2)




        elif figure_option == "Network Plots which Represents the NPDC":
            network_npdc(nodes_2, colors_2, sizes_2, edges_2)


elif analysis_option == 'National':
    countries = ['America']
    energy_option = st.selectbox('Energy Options', countries)

    if energy_option == 'America':
        figures = ['The Total Connectedness Index', "The Net Total Directional Connectedness",
                   "The Net Pairwise Directional Connectedness", "Network Plots which Represents the PCI",
                   "Network Plots which Represents the NPDC"]
        figure_option = st.selectbox('Please select a pattern', figures)
        st.markdown("<center><h1>America system</h1></center>", unsafe_allow_html=True)
        if figure_option == "The Total Connectedness Index":
            plot_tci('../dataset/tci_3.csv')


        elif figure_option == "The Net Total Directional Connectedness":
            plot_net('../dataset/net_3.csv')



        elif figure_option == "The Net Pairwise Directional Connectedness":
            plot_npdc('../dataset/npdc_3.csv')



        elif figure_option == "Network Plots which Represents the PCI":
            network_pci(nodes_data_3, edges_data_3)



        elif figure_option == "Network Plots which Represents the NPDC":
            network_npdc(nodes_3, colors_3, sizes_3, edges_3)

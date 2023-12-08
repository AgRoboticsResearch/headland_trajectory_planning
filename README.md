# Headland_trajectory_planning

This repo introduces a method of headland space modeling and trajectory planning for autonomous agricultural vehicles (AAV) operating in the orchard. The environment is modeled with geometric shapes, and the headland turning is solved similarly to the valet parking problem in a cluttered environment. We leverage the [OBCA](https://github.com/XiaojingGeorgeZhang/OBCA) trajectory planner which has been applied in [Apollo-Baidu](https://github.com/ApolloAuto/apollo). 

We applied Python to leverage the convenience of Jupyter-Notebook to introduce the algorithm in detail and have optimized the Python code thoroughly to have it run efficiently. In our algorithm, we used [Casadi](https://github.com/casadi/casadi) to solve the modeled nonlinear optimization problem.

## Updates

* **Dec 8, 2023** - Initial release.

## About

If our repo helps your project from either industry or academia, please cite our paper and star our repo. Besides, you are welcome to contact us for a discussion in agricultural robotics!!

__Author__: [Chen Peng](https://hic.zju.edu.cn/2023/0904/c72951a2797324/page.htm), [Peng Wei](https://alexwei92.github.io/), [Zhenghao Fei](https://hic.zju.edu.cn/2023/0904/c72951a2797279/page.htm), Yuankai Zhu, and [Stavros G. Vougioukas](https://faculty.engineering.ucdavis.edu/vougioukas/)

__Paper__: [Optimization-Based Motion Planning for Autonomous Agricultural Vehicles Turning in Constrained Headlands](https://arxiv.org/abs/2308.01117), Chen Peng*, Peng Wei*, Zhenghao Fei, Yuankai Zhu and Stavros G. Vougioukas (submitted to Journal of Field Robotics)

## How to run

- Install the python dependency
```bash
pip install -r requirments.txt
```
- Run the code in [Jupyter Notebook](https://github.com/AgRoboticsResearch/headland_trajectory_planning/test) step by step.

## Repo Features

- Modeling of the headland space in geometric shapes
<p align="center">
    <img src="misc/headland-model.png" />
</p>

- Classic planner headland turning results
<p align="center">
    <img src="misc/Fish-Tail-Turn.png" />
    <img src="misc/Circle-Back-Turn.png" />
    <img src="misc/Omega-Turn.png" />
</p>

- Proposed headland turning results (with implement)
<p align="center">
   <img src="misc/mower-plan.png" width="500" />
</p>

## Supplementary
- Tested on a real robot within the vineyard at UC Davis. See our [YouTube](https://www.youtube.com/watch?v=sf0uDFwpSfo) video.
<a href="https://www.youtube.com/watch?v=sf0uDFwpSfo" target="blank">
    <p align="center">
        <img src="misc/video-cover.png" width="600" height="337" />
    </p>
</a>

For any technical issues, please contact Chen Peng (penchen@ucdavis.edu) or Peng Wei (penwei@ucdavis.edu).

For commercial inquiries, please contact Chen Peng (chen.peng@zju.edu.cn) or Zhenghao Fei (zfei@zju.edu.cn).

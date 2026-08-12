import argparse
from loguru import logger
from tqdm import tqdm
from factor_engine import FactorEngine


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    args=parser.parse_args()

    logger.info('启动 GM3 AI 因子发现系统')
    engine=FactorEngine(args.data)
    logger.info('加载K线数据完成')

    for step in tqdm(range(5),desc='AI搜索进度'):
        engine.search_step(step)

    result=engine.evaluate()
    logger.info(f'最佳因子IC={result["ic"]:.4f}')
    logger.info(result)


if __name__=='__main__':
    main()

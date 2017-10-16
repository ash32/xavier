import logging
import yaml
import requests
import os
import click
import re
import pandas as pd
import functools
from datetime import datetime


@click.group(chain=True, invoke_without_command=True)
def cli():
    pass


def get_config():
    path = os.path.join(os.path.dirname(__file__), 'poloniex.yaml')
    return yaml.load(open(path))['poloniex']


@cli.command()
@click.argument('raw_dir', type=click.Path())
def get_raw_data(raw_dir):
    """ Uses config data from poloniex.yaml to download json files.
    """
    logger = logging.getLogger(__name__)
    conf = get_config()

    base_url = conf['base_url']
    url_params = conf['url_params']
    pairs = conf['btc_pairs']

    for pair in pairs:
        logger.info('Retrieving {} pair..'.format(pair))
        url_params['currencyPair'] = pair

        with open(os.path.join(raw_dir, '{}.json'.format(pair)), 'w') as f:
            json_text = requests.get(base_url, params=url_params).text
            f.write(json_text)


def get_coin_symbol(filename):
    return re.match('.+_(.+).json', filename).groups()[0]


def create_data_frame(directory, filename):
    cols = ['close', 'high', 'low']
    # dtypes = dict([cols, ])
    symbol = get_coin_symbol(filename)

    df = pd.read_json(os.path.join(directory, filename))
    df = df.set_index('date')
    df = df[cols]
    df = df.rename(columns={c: c + '_' + symbol for c in cols})
    return df


@cli.command()
@click.argument('raw_dir', type=click.Path())
@click.argument('processed_dir', type=click.Path())
def process_data(raw_dir, processed_dir):
    """ Process raw files into pricing data frame
    """
    logger = logging.getLogger(__name__)
    conf = get_config()

    start = conf['url_params']['start']*1e9
    end = datetime.utcnow()
    end = conf['url_params']['end']*1e9 if end.timestamp() > conf['url_params']['end'] else end

    dates = pd.date_range(start=start, end=end, freq='30min')
    dates = pd.DataFrame(dates).set_index(0)

    files = os.listdir(raw_dir)
    dfs = [create_data_frame(raw_dir, f) for f in files]
    bigdf = functools.reduce(lambda x, y: x.join(y), dfs, dates)
    bigdf = bigdf.fillna(method='bfill')
    bigdf = bigdf.dropna(how='all')  # Drop NA rows from the end (in case datetime.now is ahead of the last row's date

    logger.info('Writing out pricing data frame..')
    bigdf.to_csv(os.path.join(processed_dir, 'prices_df.csv'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    cli()


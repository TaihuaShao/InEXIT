from __future__ import print_function

import argparse

def JRM_opts(parser):
    """
    These options are passed to the construction of the model.
    """

    group = parser.add_argument_group('JRM init')
    group.add_argument('-token', type=str, default='IEMaI',
                       choices=['MvHGAT', 'RNN', 'CNN'],
                       help="""none""")
    group.add_argument('-data_path', default="data",
                       help="Source directory for data files.")
    group.add_argument('-bert_path', default="bert-base-chinese",
                       help="Source directory for bert files.")
    group.add_argument('-save_path', default="save",
                       help="Directory for saving model files.")
    group.add_argument('-log_dir', default="log",
                       help="Directory for saving log files.")
    group.add_argument('-gpu', type=int, default=0,
                       help="Device to run on")

    group = parser.add_argument_group('Training setup')

    group.add_argument('-train_batch_size', type=int, default=16,
                       help='Batch size for training')
    group.add_argument('-train_num_workers', type=int, default=8,
                       help='Num workers for training')
    group.add_argument('-valid_batch_size', type=int, default=16,
                       help='Batch size for validing')
    group.add_argument('-valid_num_workers', type=int, default=8,
                       help='Num workers for validing')
    group.add_argument('-test_batch_size', type=int, default=16,
                       help='Batch size for testing')
    group.add_argument('-test_num_workers', type=int, default=8,
                       help='Num workers for testing')

    group.add_argument('-learning_rate', type=float, default=5e-6,
                       help="""Starting learning rate.
                       Recommended settings: sgd = 1, adagrad = 0.1,
                       adadelta = 1, adam = 0.001""")
    group.add_argument('-weight_decay', type=float, default=1e-9,
                       help="""If update_learning_rate, decay learning rate by
                       this much if (i) perplexity does not decrease on the
                       validation set or (ii) steps have gone past
                       start_decay_steps""")
    group.add_argument('-max_gradient_norm', type=float, default=10.0,
                       help="""max_gradient_norm""")
    group.add_argument('-geek_max_feature_num', type=int, default=12,
                       help='max_len of geek_max_feature_len.') 
    group.add_argument('-job_max_feature_num', type=int, default=11,
                       help='max_len of job_max_feature_len.') 
    group.add_argument('-max_feat_len', type=int, default=16,
                       help='max_feat_len')
    group.add_argument('-max_sent_len', type=int, default=256,
                       help='max_sent_len')    
    group.add_argument('-word_emb_dim', type=int, default=768,
                       help='dim of word embedding.') 
    group.add_argument('-hidden_size', type=int, default=768,
                       help="hidden_size in linear layer")
    group.add_argument('-num_heads', type=int, default=8,
                       help='num_heads of Transformer.')
    group.add_argument('-num_layers', type=int, default=1,
                       help='num_layers of Transformer.')
    group.add_argument('-dropout', type=float, default=0.1,
                       help="""dropout of Transformer.""")
    group.add_argument('-fusion', type=str, default='add',
                       choices=['cat', 'add'],
                       help="""none""")

    group.add_argument('-seed', type=int, default=4,
                       help="Random seed")    
    group.add_argument('-num_epochs', type=int, default=20,
                       help='Number of training epochs.')
    group.add_argument('-end_step', type=int, default=5,
                       help='Early stop when decrease for this epochs.')
    group.add_argument('-resume', action="store_true",
                       help='Whether to train from the checkpoint model.')


def add_md_help_argument(parser):
    """ md help parser """
    parser.add_argument('-md', action=MarkdownHelpAction,
                        help='print Markdown-formatted help text and exit.')


# MARKDOWN boilerplate

# Copyright 2016 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
class MarkdownHelpFormatter(argparse.HelpFormatter):
    """A really bare-bones argparse help formatter that generates valid markdown.
    This will generate something like:
    usage
    # **section heading**:
    ## **--argument-one**
    ```
    argument-one help text
    ```
    """

    def _format_usage(self, usage, actions, groups, prefix):
        return ""

    def format_help(self):
        print(self._prog)
        self._root_section.heading = '# Options: %s' % self._prog
        return super(MarkdownHelpFormatter, self).format_help()

    def start_section(self, heading):
        super(MarkdownHelpFormatter, self) \
            .start_section('### **%s**' % heading)

    def _format_action(self, action):
        if action.dest == "help" or action.dest == "md":
            return ""
        lines = []
        lines.append('* **-%s %s** ' % (action.dest,
                                        "[%s]" % action.default
                                        if action.default else "[]"))
        if action.help:
            help_text = self._expand_help(action)
            lines.extend(self._split_lines(help_text, 80))
        lines.extend(['', ''])
        return '\n'.join(lines)


class MarkdownHelpAction(argparse.Action):
    """ MD help action """

    def __init__(self, option_strings,
                 dest=argparse.SUPPRESS, default=argparse.SUPPRESS,
                 **kwargs):
        super(MarkdownHelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.formatter_class = MarkdownHelpFormatter
        parser.print_help()
        parser.exit()


class DeprecateAction(argparse.Action):
    """ Deprecate action """

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(DeprecateAction, self).__init__(option_strings, dest, nargs=0,
                                              help=help, **kwargs)

    def __call__(self, parser, namespace, values, flag_name):
        help = self.help if self.mdhelp is not None else ""
        msg = "Flag '%s' is deprecated. %s" % (flag_name, help)
        raise argparse.ArgumentTypeError(msg)

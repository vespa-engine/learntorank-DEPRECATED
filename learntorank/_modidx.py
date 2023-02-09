# Autogenerated by nbdev

d = { 'settings': { 'branch': 'main',
                'doc_baseurl': '/learntorank/',
                'doc_host': 'https://thigm85.github.io',
                'git_url': 'https://github.com/vespa-engine/learntorank/',
                'lib_path': 'learntorank'},
  'syms': { 'learntorank.evaluation': { 'learntorank.evaluation.EvalMetric': ( 'module_evaluation.html#evalmetric',
                                                                               'learntorank/evaluation.py'),
                                        'learntorank.evaluation.EvalMetric.__init__': ( 'module_evaluation.html#evalmetric.__init__',
                                                                                        'learntorank/evaluation.py'),
                                        'learntorank.evaluation.EvalMetric.evaluate_query': ( 'module_evaluation.html#evalmetric.evaluate_query',
                                                                                              'learntorank/evaluation.py'),
                                        'learntorank.evaluation.MatchRatio': ( 'module_evaluation.html#matchratio',
                                                                               'learntorank/evaluation.py'),
                                        'learntorank.evaluation.MatchRatio.__init__': ( 'module_evaluation.html#matchratio.__init__',
                                                                                        'learntorank/evaluation.py'),
                                        'learntorank.evaluation.MatchRatio.evaluate_query': ( 'module_evaluation.html#matchratio.evaluate_query',
                                                                                              'learntorank/evaluation.py'),
                                        'learntorank.evaluation.NormalizedDiscountedCumulativeGain': ( 'module_evaluation.html#normalizeddiscountedcumulativegain',
                                                                                                       'learntorank/evaluation.py'),
                                        'learntorank.evaluation.NormalizedDiscountedCumulativeGain.__init__': ( 'module_evaluation.html#normalizeddiscountedcumulativegain.__init__',
                                                                                                                'learntorank/evaluation.py'),
                                        'learntorank.evaluation.NormalizedDiscountedCumulativeGain._compute_dcg': ( 'module_evaluation.html#normalizeddiscountedcumulativegain._compute_dcg',
                                                                                                                    'learntorank/evaluation.py'),
                                        'learntorank.evaluation.NormalizedDiscountedCumulativeGain.evaluate_query': ( 'module_evaluation.html#normalizeddiscountedcumulativegain.evaluate_query',
                                                                                                                      'learntorank/evaluation.py'),
                                        'learntorank.evaluation.Recall': ('module_evaluation.html#recall', 'learntorank/evaluation.py'),
                                        'learntorank.evaluation.Recall.__init__': ( 'module_evaluation.html#recall.__init__',
                                                                                    'learntorank/evaluation.py'),
                                        'learntorank.evaluation.Recall.evaluate_query': ( 'module_evaluation.html#recall.evaluate_query',
                                                                                          'learntorank/evaluation.py'),
                                        'learntorank.evaluation.ReciprocalRank': ( 'module_evaluation.html#reciprocalrank',
                                                                                   'learntorank/evaluation.py'),
                                        'learntorank.evaluation.ReciprocalRank.__init__': ( 'module_evaluation.html#reciprocalrank.__init__',
                                                                                            'learntorank/evaluation.py'),
                                        'learntorank.evaluation.ReciprocalRank.evaluate_query': ( 'module_evaluation.html#reciprocalrank.evaluate_query',
                                                                                                  'learntorank/evaluation.py'),
                                        'learntorank.evaluation.TimeQuery': ( 'module_evaluation.html#timequery',
                                                                              'learntorank/evaluation.py'),
                                        'learntorank.evaluation.TimeQuery.__init__': ( 'module_evaluation.html#timequery.__init__',
                                                                                       'learntorank/evaluation.py'),
                                        'learntorank.evaluation.TimeQuery.evaluate_query': ( 'module_evaluation.html#timequery.evaluate_query',
                                                                                             'learntorank/evaluation.py'),
                                        'learntorank.evaluation._evaluate_query_retry': ( 'module_evaluation.html#_evaluate_query_retry',
                                                                                          'learntorank/evaluation.py'),
                                        'learntorank.evaluation.evaluate': ('module_evaluation.html#evaluate', 'learntorank/evaluation.py'),
                                        'learntorank.evaluation.evaluate_query': ( 'module_evaluation.html#evaluate_query',
                                                                                   'learntorank/evaluation.py')},
            'learntorank.ml': { 'learntorank.ml.BertModelConfig': ('module_ml.html#bertmodelconfig', 'learntorank/ml.py'),
                                'learntorank.ml.BertModelConfig.__eq__': ('module_ml.html#bertmodelconfig.__eq__', 'learntorank/ml.py'),
                                'learntorank.ml.BertModelConfig.__init__': ('module_ml.html#bertmodelconfig.__init__', 'learntorank/ml.py'),
                                'learntorank.ml.BertModelConfig.__repr__': ('module_ml.html#bertmodelconfig.__repr__', 'learntorank/ml.py'),
                                'learntorank.ml.BertModelConfig._doc_input_ids': ( 'module_ml.html#bertmodelconfig._doc_input_ids',
                                                                                   'learntorank/ml.py'),
                                'learntorank.ml.BertModelConfig._generate_dummy_inputs': ( 'module_ml.html#bertmodelconfig._generate_dummy_inputs',
                                                                                           'learntorank/ml.py'),
                                'learntorank.ml.BertModelConfig._query_input_ids': ( 'module_ml.html#bertmodelconfig._query_input_ids',
                                                                                     'learntorank/ml.py'),
                                'learntorank.ml.BertModelConfig._validate_model': ( 'module_ml.html#bertmodelconfig._validate_model',
                                                                                    'learntorank/ml.py'),
                                'learntorank.ml.BertModelConfig._validate_tokenizer': ( 'module_ml.html#bertmodelconfig._validate_tokenizer',
                                                                                        'learntorank/ml.py'),
                                'learntorank.ml.BertModelConfig.add_model': ( 'module_ml.html#bertmodelconfig.add_model',
                                                                              'learntorank/ml.py'),
                                'learntorank.ml.BertModelConfig.create_encodings': ( 'module_ml.html#bertmodelconfig.create_encodings',
                                                                                     'learntorank/ml.py'),
                                'learntorank.ml.BertModelConfig.doc_fields': ( 'module_ml.html#bertmodelconfig.doc_fields',
                                                                               'learntorank/ml.py'),
                                'learntorank.ml.BertModelConfig.document_fields': ( 'module_ml.html#bertmodelconfig.document_fields',
                                                                                    'learntorank/ml.py'),
                                'learntorank.ml.BertModelConfig.export_to_onnx': ( 'module_ml.html#bertmodelconfig.export_to_onnx',
                                                                                   'learntorank/ml.py'),
                                'learntorank.ml.BertModelConfig.onnx_model': ( 'module_ml.html#bertmodelconfig.onnx_model',
                                                                               'learntorank/ml.py'),
                                'learntorank.ml.BertModelConfig.predict': ('module_ml.html#bertmodelconfig.predict', 'learntorank/ml.py'),
                                'learntorank.ml.BertModelConfig.query_profile_type_fields': ( 'module_ml.html#bertmodelconfig.query_profile_type_fields',
                                                                                              'learntorank/ml.py'),
                                'learntorank.ml.BertModelConfig.query_tensor_mapping': ( 'module_ml.html#bertmodelconfig.query_tensor_mapping',
                                                                                         'learntorank/ml.py'),
                                'learntorank.ml.BertModelConfig.rank_profile': ( 'module_ml.html#bertmodelconfig.rank_profile',
                                                                                 'learntorank/ml.py'),
                                'learntorank.ml.ModelConfig': ('module_ml.html#modelconfig', 'learntorank/ml.py'),
                                'learntorank.ml.ModelConfig.__init__': ('module_ml.html#modelconfig.__init__', 'learntorank/ml.py'),
                                'learntorank.ml.ModelConfig.document_fields': ( 'module_ml.html#modelconfig.document_fields',
                                                                                'learntorank/ml.py'),
                                'learntorank.ml.ModelConfig.onnx_model': ('module_ml.html#modelconfig.onnx_model', 'learntorank/ml.py'),
                                'learntorank.ml.ModelConfig.query_profile_type_fields': ( 'module_ml.html#modelconfig.query_profile_type_fields',
                                                                                          'learntorank/ml.py'),
                                'learntorank.ml.ModelConfig.rank_profile': ('module_ml.html#modelconfig.rank_profile', 'learntorank/ml.py'),
                                'learntorank.ml.ModelServer': ('module_ml.html#modelserver', 'learntorank/ml.py'),
                                'learntorank.ml.ModelServer.__init__': ('module_ml.html#modelserver.__init__', 'learntorank/ml.py'),
                                'learntorank.ml.ModelServer.from_dict': ('module_ml.html#modelserver.from_dict', 'learntorank/ml.py'),
                                'learntorank.ml.ModelServer.to_dict': ('module_ml.html#modelserver.to_dict', 'learntorank/ml.py'),
                                'learntorank.ml.SequenceClassification': ('module_ml.html#sequenceclassification', 'learntorank/ml.py'),
                                'learntorank.ml.SequenceClassification.__init__': ( 'module_ml.html#sequenceclassification.__init__',
                                                                                    'learntorank/ml.py'),
                                'learntorank.ml.SequenceClassification.create_url_encoded_tokens': ( 'module_ml.html#sequenceclassification.create_url_encoded_tokens',
                                                                                                     'learntorank/ml.py'),
                                'learntorank.ml.Task': ('module_ml.html#task', 'learntorank/ml.py'),
                                'learntorank.ml.Task.__init__': ('module_ml.html#task.__init__', 'learntorank/ml.py'),
                                'learntorank.ml.TextTask': ('module_ml.html#texttask', 'learntorank/ml.py'),
                                'learntorank.ml.TextTask.__init__': ('module_ml.html#texttask.__init__', 'learntorank/ml.py'),
                                'learntorank.ml.TextTask._create_pipeline': ( 'module_ml.html#texttask._create_pipeline',
                                                                              'learntorank/ml.py'),
                                'learntorank.ml.TextTask._load_tokenizer': ('module_ml.html#texttask._load_tokenizer', 'learntorank/ml.py'),
                                'learntorank.ml.TextTask.create_url_encoded_tokens': ( 'module_ml.html#texttask.create_url_encoded_tokens',
                                                                                       'learntorank/ml.py'),
                                'learntorank.ml.TextTask.export_to_onnx': ('module_ml.html#texttask.export_to_onnx', 'learntorank/ml.py'),
                                'learntorank.ml.TextTask.parse_vespa_prediction': ( 'module_ml.html#texttask.parse_vespa_prediction',
                                                                                    'learntorank/ml.py'),
                                'learntorank.ml.TextTask.predict': ('module_ml.html#texttask.predict', 'learntorank/ml.py'),
                                'learntorank.ml.add_ranking_model': ('module_ml.html#add_ranking_model', 'learntorank/ml.py')},
            'learntorank.passage': { 'learntorank.passage.PassageData': ('module_passage.html#passagedata', 'learntorank/passage.py'),
                                     'learntorank.passage.PassageData.__eq__': ( 'module_passage.html#passagedata.__eq__',
                                                                                 'learntorank/passage.py'),
                                     'learntorank.passage.PassageData.__init__': ( 'module_passage.html#passagedata.__init__',
                                                                                   'learntorank/passage.py'),
                                     'learntorank.passage.PassageData.__repr__': ( 'module_passage.html#passagedata.__repr__',
                                                                                   'learntorank/passage.py'),
                                     'learntorank.passage.PassageData.get_corpus': ( 'module_passage.html#passagedata.get_corpus',
                                                                                     'learntorank/passage.py'),
                                     'learntorank.passage.PassageData.get_labels': ( 'module_passage.html#passagedata.get_labels',
                                                                                     'learntorank/passage.py'),
                                     'learntorank.passage.PassageData.get_queries': ( 'module_passage.html#passagedata.get_queries',
                                                                                      'learntorank/passage.py'),
                                     'learntorank.passage.PassageData.load': ( 'module_passage.html#passagedata.load',
                                                                               'learntorank/passage.py'),
                                     'learntorank.passage.PassageData.save': ( 'module_passage.html#passagedata.save',
                                                                               'learntorank/passage.py'),
                                     'learntorank.passage.PassageData.summary': ( 'module_passage.html#passagedata.summary',
                                                                                  'learntorank/passage.py'),
                                     'learntorank.passage.create_basic_search_package': ( 'module_passage.html#create_basic_search_package',
                                                                                          'learntorank/passage.py'),
                                     'learntorank.passage.evaluate_query_models': ( 'module_passage.html#evaluate_query_models',
                                                                                    'learntorank/passage.py'),
                                     'learntorank.passage.load_data': ('module_passage.html#load_data', 'learntorank/passage.py'),
                                     'learntorank.passage.sample_data': ('module_passage.html#sample_data', 'learntorank/passage.py'),
                                     'learntorank.passage.sample_dict_items': ( 'module_passage.html#sample_dict_items',
                                                                                'learntorank/passage.py'),
                                     'learntorank.passage.save_data': ('module_passage.html#save_data', 'learntorank/passage.py')},
            'learntorank.query': { 'learntorank.query.AND': ('module_query.html#and', 'learntorank/query.py'),
                                   'learntorank.query.AND.__init__': ('module_query.html#and.__init__', 'learntorank/query.py'),
                                   'learntorank.query.AND.create_match_filter': ( 'module_query.html#and.create_match_filter',
                                                                                  'learntorank/query.py'),
                                   'learntorank.query.AND.get_query_properties': ( 'module_query.html#and.get_query_properties',
                                                                                   'learntorank/query.py'),
                                   'learntorank.query.ANN': ('module_query.html#ann', 'learntorank/query.py'),
                                   'learntorank.query.ANN.__init__': ('module_query.html#ann.__init__', 'learntorank/query.py'),
                                   'learntorank.query.ANN.create_match_filter': ( 'module_query.html#ann.create_match_filter',
                                                                                  'learntorank/query.py'),
                                   'learntorank.query.ANN.get_query_properties': ( 'module_query.html#ann.get_query_properties',
                                                                                   'learntorank/query.py'),
                                   'learntorank.query.MatchFilter': ('module_query.html#matchfilter', 'learntorank/query.py'),
                                   'learntorank.query.MatchFilter.__init__': ( 'module_query.html#matchfilter.__init__',
                                                                               'learntorank/query.py'),
                                   'learntorank.query.MatchFilter.create_match_filter': ( 'module_query.html#matchfilter.create_match_filter',
                                                                                          'learntorank/query.py'),
                                   'learntorank.query.MatchFilter.get_query_properties': ( 'module_query.html#matchfilter.get_query_properties',
                                                                                           'learntorank/query.py'),
                                   'learntorank.query.OR': ('module_query.html#or', 'learntorank/query.py'),
                                   'learntorank.query.OR.__init__': ('module_query.html#or.__init__', 'learntorank/query.py'),
                                   'learntorank.query.OR.create_match_filter': ( 'module_query.html#or.create_match_filter',
                                                                                 'learntorank/query.py'),
                                   'learntorank.query.OR.get_query_properties': ( 'module_query.html#or.get_query_properties',
                                                                                  'learntorank/query.py'),
                                   'learntorank.query.QueryModel': ('module_query.html#querymodel', 'learntorank/query.py'),
                                   'learntorank.query.QueryModel.__init__': ( 'module_query.html#querymodel.__init__',
                                                                              'learntorank/query.py'),
                                   'learntorank.query.QueryModel.create_body': ( 'module_query.html#querymodel.create_body',
                                                                                 'learntorank/query.py'),
                                   'learntorank.query.QueryProperty': ('module_query.html#queryproperty', 'learntorank/query.py'),
                                   'learntorank.query.QueryProperty.__init__': ( 'module_query.html#queryproperty.__init__',
                                                                                 'learntorank/query.py'),
                                   'learntorank.query.QueryProperty.get_query_properties': ( 'module_query.html#queryproperty.get_query_properties',
                                                                                             'learntorank/query.py'),
                                   'learntorank.query.QueryRankingFeature': ( 'module_query.html#queryrankingfeature',
                                                                              'learntorank/query.py'),
                                   'learntorank.query.QueryRankingFeature.__init__': ( 'module_query.html#queryrankingfeature.__init__',
                                                                                       'learntorank/query.py'),
                                   'learntorank.query.QueryRankingFeature.get_query_properties': ( 'module_query.html#queryrankingfeature.get_query_properties',
                                                                                                   'learntorank/query.py'),
                                   'learntorank.query.Ranking': ('module_query.html#ranking', 'learntorank/query.py'),
                                   'learntorank.query.Ranking.__init__': ('module_query.html#ranking.__init__', 'learntorank/query.py'),
                                   'learntorank.query.Tokenize': ('module_query.html#tokenize', 'learntorank/query.py'),
                                   'learntorank.query.Tokenize.__init__': ('module_query.html#tokenize.__init__', 'learntorank/query.py'),
                                   'learntorank.query.Tokenize.create_match_filter': ( 'module_query.html#tokenize.create_match_filter',
                                                                                       'learntorank/query.py'),
                                   'learntorank.query.Tokenize.get_query_properties': ( 'module_query.html#tokenize.get_query_properties',
                                                                                        'learntorank/query.py'),
                                   'learntorank.query.Union': ('module_query.html#union', 'learntorank/query.py'),
                                   'learntorank.query.Union.__init__': ('module_query.html#union.__init__', 'learntorank/query.py'),
                                   'learntorank.query.Union.create_match_filter': ( 'module_query.html#union.create_match_filter',
                                                                                    'learntorank/query.py'),
                                   'learntorank.query.Union.get_query_properties': ( 'module_query.html#union.get_query_properties',
                                                                                     'learntorank/query.py'),
                                   'learntorank.query.WeakAnd': ('module_query.html#weakand', 'learntorank/query.py'),
                                   'learntorank.query.WeakAnd.__init__': ('module_query.html#weakand.__init__', 'learntorank/query.py'),
                                   'learntorank.query.WeakAnd.create_match_filter': ( 'module_query.html#weakand.create_match_filter',
                                                                                      'learntorank/query.py'),
                                   'learntorank.query.WeakAnd.get_query_properties': ( 'module_query.html#weakand.get_query_properties',
                                                                                       'learntorank/query.py'),
                                   'learntorank.query._annotate_data': ('module_query.html#_annotate_data', 'learntorank/query.py'),
                                   'learntorank.query._build_query_body': ('module_query.html#_build_query_body', 'learntorank/query.py'),
                                   'learntorank.query._parse_labeled_data': ( 'module_query.html#_parse_labeled_data',
                                                                              'learntorank/query.py'),
                                   'learntorank.query.collect_vespa_features': ( 'module_query.html#collect_vespa_features',
                                                                                 'learntorank/query.py'),
                                   'learntorank.query.send_query': ('module_query.html#send_query', 'learntorank/query.py'),
                                   'learntorank.query.send_query_batch': ('module_query.html#send_query_batch', 'learntorank/query.py'),
                                   'learntorank.query.store_vespa_features': ( 'module_query.html#store_vespa_features',
                                                                               'learntorank/query.py')},
            'learntorank.ranking': { 'learntorank.ranking.LassoHyperModel': ( 'module_ranking.html#lassohypermodel',
                                                                              'learntorank/ranking.py'),
                                     'learntorank.ranking.LassoHyperModel.__init__': ( 'module_ranking.html#lassohypermodel.__init__',
                                                                                       'learntorank/ranking.py'),
                                     'learntorank.ranking.LassoHyperModel.build': ( 'module_ranking.html#lassohypermodel.build',
                                                                                    'learntorank/ranking.py'),
                                     'learntorank.ranking.LinearHyperModel': ( 'module_ranking.html#linearhypermodel',
                                                                               'learntorank/ranking.py'),
                                     'learntorank.ranking.LinearHyperModel.__init__': ( 'module_ranking.html#linearhypermodel.__init__',
                                                                                        'learntorank/ranking.py'),
                                     'learntorank.ranking.LinearHyperModel.build': ( 'module_ranking.html#linearhypermodel.build',
                                                                                     'learntorank/ranking.py'),
                                     'learntorank.ranking.ListwiseRankingFramework': ( 'module_ranking.html#listwiserankingframework',
                                                                                       'learntorank/ranking.py'),
                                     'learntorank.ranking.ListwiseRankingFramework.__init__': ( 'module_ranking.html#listwiserankingframework.__init__',
                                                                                                'learntorank/ranking.py'),
                                     'learntorank.ranking.ListwiseRankingFramework._forward_selection_iteration': ( 'module_ranking.html#listwiserankingframework._forward_selection_iteration',
                                                                                                                    'learntorank/ranking.py'),
                                     'learntorank.ranking.ListwiseRankingFramework.create_and_train_normalization_layer': ( 'module_ranking.html#listwiserankingframework.create_and_train_normalization_layer',
                                                                                                                            'learntorank/ranking.py'),
                                     'learntorank.ranking.ListwiseRankingFramework.create_dataset': ( 'module_ranking.html#listwiserankingframework.create_dataset',
                                                                                                      'learntorank/ranking.py'),
                                     'learntorank.ranking.ListwiseRankingFramework.fit_lasso_linear_model': ( 'module_ranking.html#listwiserankingframework.fit_lasso_linear_model',
                                                                                                              'learntorank/ranking.py'),
                                     'learntorank.ranking.ListwiseRankingFramework.fit_linear_model': ( 'module_ranking.html#listwiserankingframework.fit_linear_model',
                                                                                                        'learntorank/ranking.py'),
                                     'learntorank.ranking.ListwiseRankingFramework.forward_selection_model_search': ( 'module_ranking.html#listwiserankingframework.forward_selection_model_search',
                                                                                                                      'learntorank/ranking.py'),
                                     'learntorank.ranking.ListwiseRankingFramework.lasso_model_search': ( 'module_ranking.html#listwiserankingframework.lasso_model_search',
                                                                                                          'learntorank/ranking.py'),
                                     'learntorank.ranking.ListwiseRankingFramework.listwise_tf_dataset_from_csv': ( 'module_ranking.html#listwiserankingframework.listwise_tf_dataset_from_csv',
                                                                                                                    'learntorank/ranking.py'),
                                     'learntorank.ranking.ListwiseRankingFramework.listwise_tf_dataset_from_df': ( 'module_ranking.html#listwiserankingframework.listwise_tf_dataset_from_df',
                                                                                                                   'learntorank/ranking.py'),
                                     'learntorank.ranking.ListwiseRankingFramework.tune_model': ( 'module_ranking.html#listwiserankingframework.tune_model',
                                                                                                  'learntorank/ranking.py'),
                                     'learntorank.ranking.keras_lasso_linear_model': ( 'module_ranking.html#keras_lasso_linear_model',
                                                                                       'learntorank/ranking.py'),
                                     'learntorank.ranking.keras_linear_model': ( 'module_ranking.html#keras_linear_model',
                                                                                 'learntorank/ranking.py'),
                                     'learntorank.ranking.keras_ndcg_compiled_model': ( 'module_ranking.html#keras_ndcg_compiled_model',
                                                                                        'learntorank/ranking.py')},
            'learntorank.stats': { 'learntorank.stats.bootstrap_sampling': ('module_stats.html#bootstrap_sampling', 'learntorank/stats.py'),
                                   'learntorank.stats.compute_evaluation_estimates': ( 'module_stats.html#compute_evaluation_estimates',
                                                                                       'learntorank/stats.py')},
            'learntorank.test_integration': {},
            'learntorank.test_integration_ranking': {},
            'learntorank.test_integration_running_instance': {},
            'learntorank.test_ml': {},
            'learntorank.test_package': {}}}

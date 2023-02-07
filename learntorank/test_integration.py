# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import unittest
import os
from vespa.deployment import VespaDocker
from vespa.package import (
    HNSW,
    Document,
    Field,
    Schema,
    FieldSet,
    SecondPhaseRanking,
    RankProfile,
    ApplicationPackage,
    QueryProfile,
    QueryProfileType,
    QueryTypeField,
)
from learntorank.query import (
    QueryModel,
    Ranking,
    OR,
    QueryRankingFeature,
    send_query,
    store_vespa_features,
)
from learntorank.ml import (
    SequenceClassification,
    BertModelConfig,
    ModelServer,
    add_ranking_model,
)

CONTAINER_STOP_TIMEOUT = 10


def create_cord19_application_package():
    app_package = ApplicationPackage(name="cord19")
    app_package.schema.add_fields(
        Field(name="id", type="string", indexing=["attribute", "summary"]),
        Field(
            name="title",
            type="string",
            indexing=["index", "summary"],
            index="enable-bm25",
        ),
    )
    app_package.schema.add_field_set(FieldSet(name="default", fields=["title"]))
    app_package.schema.add_rank_profile(
        RankProfile(name="bm25", first_phase="bm25(title)")
    )
    bert_config = BertModelConfig(
        model_id="pretrained_bert_tiny",
        tokenizer="google/bert_uncased_L-2_H-128_A-2",
        model="google/bert_uncased_L-2_H-128_A-2",
        query_input_size=5,
        doc_input_size=10,
    )
    add_ranking_model(
        app_package=app_package,
        model_config=bert_config,
        include_model_summary_features=True,
        inherits="default",
        first_phase="bm25(title)",
        second_phase=SecondPhaseRanking(rerank_count=10, expression="logit1"),
    )
    return app_package


class TestApplicationCommon(unittest.TestCase):

    @staticmethod
    def _parse_vespa_tensor(hit, feature):
        return hit["fields"]["summaryfeatures"][feature]["values"][0]

    def bert_model_input_and_output(
            self, app, schema_name, fields_to_send, model_config
    ):
        #
        # Feed a data point
        #
        response = app.feed_data_point(
            schema=schema_name,
            data_id=fields_to_send["id"],
            fields=fields_to_send,
        )
        self.assertEqual(
            response.json["id"],
            "id:{}:{}::{}".format(schema_name, schema_name, fields_to_send["id"]),
        )
        #
        # Run a test query
        #
        result = send_query(
            app=app,
            query="this is a test",
            query_model=QueryModel(
                query_properties=[
                    QueryRankingFeature(
                        name=model_config.query_token_ids_name,
                        mapping=model_config.query_tensor_mapping,
                    )
                ],
                match_phase=OR(),
                ranking=Ranking(name="pretrained_bert_tiny"),
            ),
        )
        vespa_input_ids = self._parse_vespa_tensor(result.hits[0], "input_ids")
        vespa_attention_mask = self._parse_vespa_tensor(
            result.hits[0], "attention_mask"
        )
        vespa_token_type_ids = self._parse_vespa_tensor(
            result.hits[0], "token_type_ids"
        )

        expected_inputs = model_config.create_encodings(
            queries=["this is a test"], docs=[fields_to_send["title"]]
        )
        self.assertEqual(vespa_input_ids, expected_inputs["input_ids"][0])
        self.assertEqual(vespa_attention_mask, expected_inputs["attention_mask"][0])
        self.assertEqual(vespa_token_type_ids, expected_inputs["token_type_ids"][0])

        expected_logits = model_config.predict(
            queries=["this is a test"], docs=[fields_to_send["title"]]
        )
        self.assertAlmostEqual(
            result.hits[0]["fields"]["summaryfeatures"]["logit0"],
            expected_logits[0][0],
            5,
        )
        self.assertAlmostEqual(
            result.hits[0]["fields"]["summaryfeatures"]["logit1"],
            expected_logits[0][1],
            5,
        )


class TestCord19Application(TestApplicationCommon):
    def setUp(self) -> None:
        self.app_package = create_cord19_application_package()
        self.vespa_docker = VespaDocker(port=8089)
        self.app = self.vespa_docker.deploy(application_package=self.app_package)
        self.model_config = self.app_package.model_configs["pretrained_bert_tiny"]
        self.fields_to_send = []
        self.expected_fields_from_get_operation = []
        for i in range(10):
            fields = {
                "id": f"{i}",
                "title": f"this is title {i}",
            }
            tensor_field_dict = self.model_config.doc_fields(text=str(fields["title"]))
            fields.update(tensor_field_dict)
            self.fields_to_send.append(fields)

            expected_fields = {
                "id": f"{i}",
                "title": f"this is title {i}",
            }
            tensor_field_values = tensor_field_dict[
                "pretrained_bert_tiny_doc_token_ids"
            ]["values"]
            expected_fields.update(
                {
                    "pretrained_bert_tiny_doc_token_ids": {
                        "type": f"tensor<float>(d0[{len(tensor_field_values)}])",
                        "values": tensor_field_values,
                    }
                }
            )
            self.expected_fields_from_get_operation.append(expected_fields)
        self.fields_to_update = [
            {
                "id": f"{i}",
                "title": "this is my updated title number {}".format(i),
            }
            for i in range(10)
        ]

    def test_bert_model_input_and_output(self):
        self.bert_model_input_and_output(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send[0],
            model_config=self.model_config,
        )

    def tearDown(self) -> None:
        self.vespa_docker.container.stop(timeout=CONTAINER_STOP_TIMEOUT)
        self.vespa_docker.container.remove()
        try:
            os.remove(os.path.join(os.environ["RESOURCES_DIR"], "vespa_features.csv"))
        except OSError:
            pass

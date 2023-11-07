# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import unittest
import os
from shutil import rmtree

from vespa.package import (
    Field,
    Document,
    FieldSet,
    SecondPhaseRanking,
    RankProfile,
    Schema,
    QueryTypeField,
    QueryField,
    ApplicationPackage,
)
from learntorank.ml import BertModelConfig, ModelServer, add_ranking_model


class TestApplicationPackageAddBertRankingWithMultipleSchemas(unittest.TestCase):
    def setUp(self) -> None:
        news_schema = Schema(
            name="news",
            document=Document(
                fields=[
                    Field(
                        name="news_id", type="string", indexing=["attribute", "summary"]
                    ),
                ]
            ),
        )
        user_schema = Schema(
            name="user",
            document=Document(
                fields=[
                    Field(
                        name="user_id", type="string", indexing=["attribute", "summary"]
                    ),
                ]
            ),
        )
        self.app_package = ApplicationPackage(
            name="testapp",
            schema=[news_schema, user_schema],
        )
        bert_config = BertModelConfig(
            model_id="bert_tiny",
            query_input_size=4,
            doc_input_size=8,
            tokenizer=os.path.join(os.environ["RESOURCES_DIR"], "bert_tiny_tokenizer"),
            model=os.path.join(os.environ["RESOURCES_DIR"], "bert_tiny_model"),
        )
        add_ranking_model(
            self.app_package,
            model_config=bert_config,
            schema="news",
            include_model_summary_features=True,
            inherits="default",
            first_phase="bm25(title)",
            second_phase=SecondPhaseRanking(rerank_count=10, expression="logit1"),
        )

        self.disk_folder = "saved_app"

    def test_news_schema_to_text(self):
        expected_result = (
            "schema news {\n"
            "    document news {\n"
            "        field news_id type string {\n"
            "            indexing: attribute | summary\n"
            "        }\n"
            "        field bert_tiny_doc_token_ids type tensor<float>(d0[7]) {\n"
            "            indexing: attribute | summary\n"
            "        }\n"
            "    }\n"
            "    onnx-model bert_tiny {\n"
            "        file: files/bert_tiny.onnx\n"
            "        input input_ids: input_ids\n"
            "        input token_type_ids: token_type_ids\n"
            "        input attention_mask: attention_mask\n"
            "        output output_0: logits\n"
            "    }\n"
            "    rank-profile bert_tiny inherits default {\n"
            "        constants {\n"
            "            TOKEN_NONE: 0\n"
            "            TOKEN_CLS: 101\n"
            "            TOKEN_SEP: 102\n"
            "        }\n"
            "        function question_length() {\n"
            "            expression {\n"
            "                sum(map(query(bert_tiny_query_token_ids), f(a)(a > 0)))\n"
            "            }\n"
            "        }\n"
            "        function doc_length() {\n"
            "            expression {\n"
            "                sum(map(attribute(bert_tiny_doc_token_ids), f(a)(a > 0)))\n"
            "            }\n"
            "        }\n"
            "        function input_ids() {\n"
            "            expression {\n"
            "                tokenInputIds(12, query(bert_tiny_query_token_ids), attribute(bert_tiny_doc_token_ids))\n"
            "            }\n"
            "        }\n"
            "        function attention_mask() {\n"
            "            expression {\n"
            "                tokenAttentionMask(12, query(bert_tiny_query_token_ids), attribute(bert_tiny_doc_token_ids))\n"
            "            }\n"
            "        }\n"
            "        function token_type_ids() {\n"
            "            expression {\n"
            "                tokenTypeIds(12, query(bert_tiny_query_token_ids), attribute(bert_tiny_doc_token_ids))\n"
            "            }\n"
            "        }\n"
            "        function logit0() {\n"
            "            expression {\n"
            "                onnx(bert_tiny).logits{d0:0,d1:0}\n"
            "            }\n"
            "        }\n"
            "        function logit1() {\n"
            "            expression {\n"
            "                onnx(bert_tiny).logits{d0:0,d1:1}\n"
            "            }\n"
            "        }\n"
            "        first-phase {\n"
            "            expression {\n"
            "                bm25(title)\n"
            "            }\n"
            "        }\n"
            "        second-phase {\n"
            "            rerank-count: 10\n"
            "            expression {\n"
            "                logit1\n"
            "            }\n"
            "        }\n"
            "        summary-features {\n"
            "            logit0\n"
            "            logit1\n"
            "            input_ids\n"
            "            attention_mask\n"
            "            token_type_ids\n"
            "        }\n"
            "    }\n"
            "}"
        )
        self.assertEqual(
            self.app_package.get_schema("news").schema_to_text, expected_result
        )

    def test_user_schema_to_text(self):
        expected_user_result = (
            "schema user {\n"
            "    document user {\n"
            "        field user_id type string {\n"
            "            indexing: attribute | summary\n"
            "        }\n"
            "    }\n"
            "}"
        )
        self.assertEqual(
            self.app_package.get_schema("user").schema_to_text, expected_user_result
        )

    def test_services_to_text(self):
        expected_result = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<services version="1.0">\n'
            '    <container id="testapp_container" version="1.0">\n'
            "        <search></search>\n"
            "        <document-api></document-api>\n"
            "    </container>\n"
            '    <content id="testapp_content" version="1.0">\n'
            '        <redundancy>1</redundancy>\n'
            "        <documents>\n"
            '            <document type="news" mode="index"></document>\n'
            '            <document type="user" mode="index"></document>\n'
            "        </documents>\n"
            "        <nodes>\n"
            '            <node distribution-key="0" hostalias="node1"></node>\n'
            "        </nodes>\n"
            "    </content>\n"
            "</services>"
        )

        self.assertEqual(self.app_package.services_to_text, expected_result)

    def test_query_profile_to_text(self):
        expected_result = (
            '<query-profile id="default" type="root">\n' "</query-profile>"
        )

        self.assertEqual(self.app_package.query_profile_to_text, expected_result)

    def test_query_profile_type_to_text(self):
        expected_result = (
            '<query-profile-type id="root">\n'
            '    <field name="ranking.features.query(bert_tiny_query_token_ids)" type="tensor&lt;float&gt;(d0[2])" />\n'
            "</query-profile-type>"
        )
        self.assertEqual(self.app_package.query_profile_type_to_text, expected_result)

    def tearDown(self) -> None:
        rmtree(self.disk_folder, ignore_errors=True)



class TestSimplifiedApplicationPackageAddBertRanking(unittest.TestCase):
    def setUp(self) -> None:
        self.app_package = ApplicationPackage(name="testapp")

        self.app_package.schema.add_fields(
            Field(name="id", type="string", indexing=["attribute", "summary"]),
            Field(
                name="title",
                type="string",
                indexing=["index", "summary"],
                index="enable-bm25",
            ),
            Field(
                name="body",
                type="string",
                indexing=["index", "summary"],
                index="enable-bm25",
            ),
        )
        self.app_package.schema.add_field_set(
            FieldSet(name="default", fields=["title", "body"])
        )
        self.app_package.schema.add_rank_profile(
            RankProfile(name="default", first_phase="nativeRank(title, body)")
        )
        self.app_package.schema.add_rank_profile(
            RankProfile(
                name="bm25",
                first_phase="bm25(title) + bm25(body)",
                inherits="default",
            )
        )
        self.app_package.query_profile_type.add_fields(
            QueryTypeField(
                name="ranking.features.query(query_bert)",
                type="tensor<float>(x[768])",
            )
        )
        self.app_package.query_profile.add_fields(
            QueryField(name="maxHits", value=100),
            QueryField(name="anotherField", value="string_value"),
        )

        bert_config = BertModelConfig(
            model_id="bert_tiny",
            query_input_size=4,
            doc_input_size=8,
            tokenizer=os.path.join(os.environ["RESOURCES_DIR"], "bert_tiny_tokenizer"),
            model=os.path.join(os.environ["RESOURCES_DIR"], "bert_tiny_model"),
        )

        add_ranking_model(
            self.app_package,
            model_config=bert_config,
            include_model_summary_features=True,
            inherits="default",
            first_phase="bm25(title)",
            second_phase=SecondPhaseRanking(rerank_count=10, expression="logit1"),
        )
        self.disk_folder = "saved_app"

    def test_schema_to_text(self):
        expected_result = (
            "schema testapp {\n"
            "    document testapp {\n"
            "        field id type string {\n"
            "            indexing: attribute | summary\n"
            "        }\n"
            "        field title type string {\n"
            "            indexing: index | summary\n"
            "            index: enable-bm25\n"
            "        }\n"
            "        field body type string {\n"
            "            indexing: index | summary\n"
            "            index: enable-bm25\n"
            "        }\n"
            "        field bert_tiny_doc_token_ids type tensor<float>(d0[7]) {\n"
            "            indexing: attribute | summary\n"
            "        }\n"
            "    }\n"
            "    fieldset default {\n"
            "        fields: title, body\n"
            "    }\n"
            "    onnx-model bert_tiny {\n"
            "        file: files/bert_tiny.onnx\n"
            "        input input_ids: input_ids\n"
            "        input token_type_ids: token_type_ids\n"
            "        input attention_mask: attention_mask\n"
            "        output output_0: logits\n"
            "    }\n"
            "    rank-profile default {\n"
            "        first-phase {\n"
            "            expression {\n"
            "                nativeRank(title, body)\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    rank-profile bm25 inherits default {\n"
            "        first-phase {\n"
            "            expression {\n"
            "                bm25(title) + bm25(body)\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    rank-profile bert_tiny inherits default {\n"
            "        constants {\n"
            "            TOKEN_NONE: 0\n"
            "            TOKEN_CLS: 101\n"
            "            TOKEN_SEP: 102\n"
            "        }\n"
            "        function question_length() {\n"
            "            expression {\n"
            "                sum(map(query(bert_tiny_query_token_ids), f(a)(a > 0)))\n"
            "            }\n"
            "        }\n"
            "        function doc_length() {\n"
            "            expression {\n"
            "                sum(map(attribute(bert_tiny_doc_token_ids), f(a)(a > 0)))\n"
            "            }\n"
            "        }\n"
            "        function input_ids() {\n"
            "            expression {\n"
            "                tokenInputIds(12, query(bert_tiny_query_token_ids), attribute(bert_tiny_doc_token_ids))\n"
            "            }\n"
            "        }\n"
            "        function attention_mask() {\n"
            "            expression {\n"
            "                tokenAttentionMask(12, query(bert_tiny_query_token_ids), attribute(bert_tiny_doc_token_ids))\n"
            "            }\n"
            "        }\n"
            "        function token_type_ids() {\n"
            "            expression {\n"
            "                tokenTypeIds(12, query(bert_tiny_query_token_ids), attribute(bert_tiny_doc_token_ids))\n"
            "            }\n"
            "        }\n"
            "        function logit0() {\n"
            "            expression {\n"
            "                onnx(bert_tiny).logits{d0:0,d1:0}\n"
            "            }\n"
            "        }\n"
            "        function logit1() {\n"
            "            expression {\n"
            "                onnx(bert_tiny).logits{d0:0,d1:1}\n"
            "            }\n"
            "        }\n"
            "        first-phase {\n"
            "            expression {\n"
            "                bm25(title)\n"
            "            }\n"
            "        }\n"
            "        second-phase {\n"
            "            rerank-count: 10\n"
            "            expression {\n"
            "                logit1\n"
            "            }\n"
            "        }\n"
            "        summary-features {\n"
            "            logit0\n"
            "            logit1\n"
            "            input_ids\n"
            "            attention_mask\n"
            "            token_type_ids\n"
            "        }\n"
            "    }\n"
            "}"
        )
        self.assertEqual(self.app_package.schema.schema_to_text, expected_result)

    def test_services_to_text(self):
        expected_result = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<services version="1.0">\n'
            '    <container id="testapp_container" version="1.0">\n'
            "        <search></search>\n"
            "        <document-api></document-api>\n"
            "    </container>\n"
            '    <content id="testapp_content" version="1.0">\n'
            '        <redundancy>1</redundancy>\n'
            "        <documents>\n"
            '            <document type="testapp" mode="index"></document>\n'
            "        </documents>\n"
            "        <nodes>\n"
            '            <node distribution-key="0" hostalias="node1"></node>\n'
            "        </nodes>\n"
            "    </content>\n"
            "</services>"
        )

        self.assertEqual(self.app_package.services_to_text, expected_result)

    def test_query_profile_to_text(self):
        expected_result = (
            '<query-profile id="default" type="root">\n'
            '    <field name="maxHits">100</field>\n'
            '    <field name="anotherField">string_value</field>\n'
            "</query-profile>"
        )

        self.assertEqual(self.app_package.query_profile_to_text, expected_result)

    def test_query_profile_type_to_text(self):
        expected_result = (
            '<query-profile-type id="root">\n'
            '    <field name="ranking.features.query(query_bert)" type="tensor&lt;float&gt;(x[768])" />\n'
            '    <field name="ranking.features.query(bert_tiny_query_token_ids)" type="tensor&lt;float&gt;(d0[2])" />\n'
            "</query-profile-type>"
        )
        self.assertEqual(self.app_package.query_profile_type_to_text, expected_result)

    def tearDown(self) -> None:
        rmtree(self.disk_folder, ignore_errors=True)


class TestModelServer(unittest.TestCase):
    def setUp(self) -> None:
        self.server_name = "testserver"
        self.model_server = ModelServer(
            name=self.server_name,
        )

    def test_model_server_serialization(self):
        self.assertEqual(
            self.model_server, ModelServer.from_dict(self.model_server.to_dict)
        )

    def test_get_schema(self):
        self.assertIsNone(self.model_server.schema)
        self.assertEqual(self.model_server.schema, self.model_server.get_schema())

    def test_services_to_text(self):
        expected_result = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<services version="1.0">\n'
            '    <container id="testserver_container" version="1.0">\n'
            "        <model-evaluation/>\n"
            "    </container>\n"
            "</services>"
        )

        self.assertEqual(self.model_server.services_to_text, expected_result)

    def test_query_profile_to_text(self):
        expected_result = (
            '<query-profile id="default" type="root">\n' "</query-profile>"
        )
        self.assertEqual(self.model_server.query_profile_to_text, expected_result)

    def test_query_profile_type_to_text(self):
        expected_result = '<query-profile-type id="root">\n' "</query-profile-type>"
        self.assertEqual(self.model_server.query_profile_type_to_text, expected_result)

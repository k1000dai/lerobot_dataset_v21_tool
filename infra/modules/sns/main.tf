# SNS Topic for triggering Lambda function
resource "aws_sns_topic" "data_pipeline_trigger" {
  name         = var.topic_name
  display_name = var.display_name

  tags = merge(var.common_tags, {
    Name = var.topic_name
  })
}

# Lambda subscription to SNS topic
resource "aws_sns_topic_subscription" "lambda_subscription" {
  topic_arn = aws_sns_topic.data_pipeline_trigger.arn
  protocol  = "lambda"
  endpoint  = var.lambda_function_arn
}

# Data source for current AWS account ID
data "aws_caller_identity" "current" {}

# Lambda permission to allow SNS to invoke the function
resource "aws_lambda_permission" "allow_sns" {
  statement_id  = "AllowExecutionFromSNS"
  action        = "lambda:InvokeFunction"
  function_name = var.lambda_function_arn
  principal     = "sns.amazonaws.com"
  source_arn    = aws_sns_topic.data_pipeline_trigger.arn
}

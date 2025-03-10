output "topic_arn" {
  description = "ARN of the SNS topic"
  value       = aws_sns_topic.data_pipeline_trigger.arn
}

output "topic_name" {
  description = "Name of the SNS topic"
  value       = aws_sns_topic.data_pipeline_trigger.name
}

output "topic_id" {
  description = "ID of the SNS topic"
  value       = aws_sns_topic.data_pipeline_trigger.id
}
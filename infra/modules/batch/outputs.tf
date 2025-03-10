output "job_queue_name" {
  description = "Name of the Batch job queue"
  value       = aws_batch_job_queue.pipeline_job_queue.name
}

output "job_definition_name" {
  description = "Name of the Batch job definition"
  value       = aws_batch_job_definition.pipeline_job_definition.name
}

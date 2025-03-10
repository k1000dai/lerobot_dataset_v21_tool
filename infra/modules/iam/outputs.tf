# Lambda IAM role outputs
output "lambda_execution_role_arn" {
  description = "ARN of the Lambda execution IAM role"
  value       = aws_iam_role.lambda_role.arn
}

output "batch_exe_role_arn" {
  description = "ARN of the Batch execution IAM role"
  value       = aws_iam_role.batch_exe_role.arn
}

output "batch_job_role_arn" {
  description = "ARN of the Batch job execution IAM role"
  value       = aws_iam_role.batch_job_role.arn
}

output "batch_computing_role_arn" {
  description = "ARN of the Batch computing IAM role"
  value       = aws_iam_role.batch_computing_role.arn
}

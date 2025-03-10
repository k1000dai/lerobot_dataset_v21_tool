# ECR Repository for rosbag2lerobot container images
resource "aws_ecr_repository" "pipeline_ecr" {
  name = "${var.environment}-${var.system_name}-ecr"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = merge(var.common_tags, {
    Name = "${var.environment}-${var.system_name}-ecr"
  })
}
